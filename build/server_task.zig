//! The server's data layer --
//! `tools/server/`'s ports of `server-common`, `server-task` and
//! `server-schema` -- its unit tests (`zig build test-server-data`) and the
//! port's half of the `server-task-diff` gate (`zig build
//! server-task-diff-bins`, into `zig-out/server-task-diff/`). Also the
//! module tree `build/tools.zig` builds `llama-server`, `llama-server-core`
//! and the tools over (`modules`, `serverModule`).
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
    /// `jinja`, for `cli/common_chat/`'s templates, which the data
    /// layer's chat conversion renders through.
    jinja: ?*std.Build.Module = null,
};

/// The target triple `--version` prints, as `build/tools.zig` formats it.
fn buildTarget(b: *std.Build, target: std.Build.ResolvedTarget) []const u8 {
    return b.fmt("{s}-{s}", .{ @tagName(target.result.cpu.arch), @tagName(target.result.os.tag) });
}

/// The modules the data layer, `llama-server`, `llama-server-core` and the
/// tools are compiled against.
pub const Modules = struct {
    tools_h: *std.Build.Module,
    tools_common: *std.Build.Module,
    cli_fit: *std.Build.Module,
    json: *std.Build.Module,
    peg: *std.Build.Module,
    common_chat: *std.Build.Module,
    mtmd: *std.Build.Module,
};

/// The module tree: `tools_common`, `cli_fit`, `json`, `peg`,
/// `common_chat` and libmtmd's port.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target, libraries and `jinja`; `opts.optimize` builds the
///   translated headers and libmtmd.
/// - `optimize`: the mode of the Zig modules themselves.
///
/// Return: the modules, owned by `b`.
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

    // `cli/fit.zig`, the port of `common/fit.cpp`, shared with the CLI.
    const fit_module = b.createModule(.{ .root_source_file = b.path("cli/fit.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    fit_module.addImport("llama", opts.reference.llama.root_module);
    fit_module.addImport("stdlib", llamacpp.stdlibOf(opts.reference.llama.root_module));

    const common_module = b.createModule(.{ .root_source_file = b.path("tools/common/module.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    common_module.addImport("tools_h", tools_h_module);
    common_module.addOptions("build_info", build_info);
    common_module.addImport("cli_fit", fit_module);
    // libllama's module, for what `common/speculative.cpp` reaches through
    // `llama-ext.h` (C++ linkage, so not in `tools_h`) and for `libcxx`.
    common_module.addImport("llama", opts.reference.llama.root_module);
    // `io.zig` writes through `stdlib.stdio`, the one instance libggml flushes.
    common_module.addImport("stdlib", llamacpp.stdlibOf(opts.reference.llama.root_module));

    // `json` is `cli_fit`'s `cli/json/` under the name `cli/common_chat/`
    // imports (`tools/server/json_module.zig` says why).
    const json_module = b.createModule(.{ .root_source_file = b.path("tools/server/json_module.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    json_module.addImport("cli_fit", fit_module);
    json_module.addImport("stdlib", llamacpp.stdlibOf(opts.reference.llama.root_module));

    // The PEG engine, which `cli/common_chat/chat_peg.zig` imports.
    const peg_module = b.createModule(.{ .root_source_file = b.path("cli/peg/module.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    peg_module.addImport("json", json_module);
    peg_module.addImport("llama", opts.reference.llama.root_module);
    peg_module.addImport("stdlib", llamacpp.stdlibOf(opts.reference.llama.root_module));

    const chat_module = b.createModule(.{ .root_source_file = b.path("cli/common_chat/module.zig"), .target = opts.target, .optimize = optimize, .link_libc = true });
    chat_module.addImport("json", json_module);
    chat_module.addImport("peg", peg_module);
    chat_module.addImport("llama", opts.reference.llama.root_module);
    chat_module.addImport("stdlib", llamacpp.stdlibOf(opts.reference.llama.root_module));
    if (opts.jinja) |j| chat_module.addImport("zigjinja", j);

    // libmtmd's port: its `exports.zig` gives the mtmd tools and the server
    // the `mtmd.h` and `mtmd-helper.h` symbols they call through `tools_h`.
    const mtmd_module = mtmd_build.module(b, .{
        .target = opts.target,
        .optimize = opts.optimize,
        .sdk = opts.sdk,
        .reference = opts.reference,
    });

    return .{
        .tools_h = tools_h_module,
        .tools_common = common_module,
        .cli_fit = fit_module,
        .json = json_module,
        .peg = peg_module,
        .common_chat = chat_module,
        .mtmd = mtmd_module,
    };
}

/// A module of `llama-server`'s files over `m`, with everything they
/// import: rooted at `tools/server/main.zig` it is the executable, at
/// `tools/server/core.zig` it is `llama-server-core`. `context.zig` reaches
/// `http.zig`, which imports `ui_assets`, so both need it.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target, mode, SDK and libllama's module.
/// - `m`: the module tree from `modules`.
/// - `root_source_file`: the module's root.
/// - `ui_assets`: the embedded web UI (`build/ui.zig`'s `port_module`).
///
/// Return: the module, owned by `b`.
pub fn serverModule(b: *std.Build, opts: Options, m: Modules, root_source_file: std.Build.LazyPath, ui_assets: *std.Build.Module) *std.Build.Module {
    const mod = b.createModule(.{
        .root_source_file = root_source_file,
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
    });
    mod.addImport("tools_common", m.tools_common);
    mod.addImport("tools_h", m.tools_h);
    mod.addImport("cli_fit", m.cli_fit);
    mod.addImport("llama", opts.reference.llama.root_module);
    mod.addImport("stdlib", llamacpp.stdlibOf(opts.reference.llama.root_module));
    mod.addImport("mtmd", m.mtmd);
    mod.addImport("ui_assets", ui_assets);
    mod.addImport("json", m.json);
    mod.addImport("common_chat", m.common_chat);
    mod.addImport("peg", m.peg);
    llamacpp.addSdkPaths(b, mod, opts.sdk);
    llamacpp.addLibgcc(b, mod);
    return mod;
}

/// Adds `test-server-data` and `server-task-diff-bins`.
pub fn add(b: *std.Build, opts: Options) void {
    // The data layer's own code is built ReleaseSafe whatever the
    // libraries under it are, as the HTTP layer's harness is.
    const safe: std.builtin.OptimizeMode = if (opts.optimize == .debug) .debug else .safe;
    const m = modules(b, opts, safe);

    // The data layer itself, rooted at `tools/server/module.zig`.
    const data = b.createModule(.{ .root_source_file = b.path("tools/server/module.zig"), .target = opts.target, .optimize = safe, .link_libc = true });
    data.addImport("tools_common", m.tools_common);
    data.addImport("tools_h", m.tools_h);
    data.addImport("cli_fit", m.cli_fit);
    data.addImport("llama", opts.reference.llama.root_module);
    data.addImport("stdlib", llamacpp.stdlibOf(opts.reference.llama.root_module));
    data.addImport("json", m.json);
    data.addImport("common_chat", m.common_chat);
    data.addImport("peg", m.peg);
    data.addImport("mtmd", m.mtmd);

    const tests = b.addTest(.{ .root_module = data });
    llamacpp.addSdkPaths(b, data, opts.sdk);
    const test_step = b.step("test-server-data", "Run the server data layer's unit tests (tools/server/common, tokens, oaicompat, task, result, schema)");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    const step = b.step("server-task-diff-bins", "Build the port's server-task-diff harness into zig-out/server-task-diff");
    const port_mod = b.createModule(.{ .root_source_file = b.path("harness/server_task_diff.zig"), .target = opts.target, .optimize = safe, .link_libc = true, .link_libcpp = true });
    port_mod.addImport("server_data", data);
    port_mod.addImport("tools_common", m.tools_common);
    port_mod.addImport("tools_h", m.tools_h);
    port_mod.addImport("llama", opts.reference.llama.root_module);
    port_mod.addImport("stdlib", llamacpp.stdlibOf(opts.reference.llama.root_module));
    port_mod.addImport("json", m.json);
    port_mod.addImport("common_chat", m.common_chat);
    port_mod.addImport("mtmd", m.mtmd);
    llamacpp.addSdkPaths(b, port_mod, opts.sdk);
    llamacpp.addLibgcc(b, port_mod);
    llamacpp.addCudaRuntime(b, port_mod, opts.cuda);
    const port_exe = b.addExecutable(.{ .name = "server_task_port", .root_module = port_mod });
    step.dependOn(&b.addInstallArtifact(port_exe, .{ .dest_dir = .{ .override = .{ .custom = "server-task-diff" } } }).step);
}
