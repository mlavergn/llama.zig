//! Upstream's tools beyond `llama-cli` (PLAN.md, Decision 61): our Zig ports
//! under `tools/`, and upstream's own, compiled unmodified, as the references
//! their gates diff against.
//!
//! **Not a port.** This is build description. The references are built as
//! `llama.cpp.zmake` builds upstream's `llama-common` (its `build.zig`,
//! `addCli`), over `libllama-ref.a` and our `libggml.a`, so the only thing that
//! differs between a port and its reference is the tool's own code -- the
//! library under both is gated already.

const std = @import("std");
const builtin = @import("builtin");
const llamacpp = @import("llamacpp.zig");
const upstream = @import("upstream.zig");
const mtmd_build = @import("mtmd.zig");

const root = "llama.cpp";

/// The pin's build number and commit, as `llama.cpp.zmake`'s `build.zig`
/// has them, so a tool's `--version` reads the same from both sides of a gate.
pub const build_number = upstream.build_number;
pub const commit = upstream.commit;

/// One upstream tool.
const Tool = struct {
    /// Upstream's directory under `tools/`, and the binary's name after
    /// `llama-`.
    name: []const u8,
    /// Our port's root, under `tools/`.
    port: []const u8,
    /// Upstream's sources, relative to `llama.cpp/`.
    ref: []const []const u8,
};

/// The tools ported so far, in the order Decision 61 takes them.
const tools = [_]Tool{
    .{ .name = "gguf-split", .port = "tools/gguf_split.zig", .ref = &.{"tools/gguf-split/gguf-split.cpp"} },
    .{ .name = "quantize", .port = "tools/quantize.zig", .ref = &.{ "tools/quantize/quantize.cpp", "tools/quantize/main.cpp" } },
    .{ .name = "tokenize", .port = "tools/tokenize.zig", .ref = &.{"tools/tokenize/tokenize.cpp"} },
    .{ .name = "fit-params", .port = "tools/fit_params.zig", .ref = &.{ "tools/fit-params/fit-params.cpp", "tools/fit-params/main.cpp" } },
    .{ .name = "export-lora", .port = "tools/export_lora.zig", .ref = &.{"tools/export-lora/export-lora.cpp"} },
    .{ .name = "batched-bench", .port = "tools/batched_bench.zig", .ref = &.{ "tools/batched-bench/batched-bench.cpp", "tools/batched-bench/main.cpp" } },
    .{ .name = "cvector-generator", .port = "tools/cvector_generator.zig", .ref = &.{"tools/cvector-generator/cvector-generator.cpp"} },
    .{ .name = "imatrix", .port = "tools/imatrix.zig", .ref = &.{"tools/imatrix/imatrix.cpp"} },
    .{ .name = "perplexity", .port = "tools/perplexity.zig", .ref = &.{ "tools/perplexity/perplexity.cpp", "tools/perplexity/main.cpp" } },
    .{ .name = "bench", .port = "tools/llama_bench.zig", .ref = &.{ "tools/llama-bench/llama-bench.cpp", "tools/llama-bench/main.cpp" } },
};

/// The tools over mtmd (Stage 8): ported, and linking the port of libmtmd
/// (`build/mtmd.zig`'s module) as their references link upstream's.
const mtmd_tools = [_]Tool{
    .{ .name = "tts", .port = "tools/tts.zig", .ref = &.{"tools/tts/tts.cpp"} },
    .{ .name = "mtmd-cli", .port = "tools/mtmd_cli.zig", .ref = &.{"tools/mtmd/mtmd-cli.cpp"} },
};

/// `llama.cpp.zmake`'s `mtmd_sources`; the per-architecture files under
/// `models/` are listed by `addMtmdRef`, as zmake globs them.
const mtmd_sources = [_][]const u8{
    "tools/mtmd/mtmd.cpp",
    "tools/mtmd/mtmd-audio.cpp",
    "tools/mtmd/mtmd-image.cpp",
    "tools/mtmd/mtmd-helper.cpp",
    "tools/mtmd/mtmd-helper-gen.cpp",
    "tools/mtmd/clip.cpp",
};

/// `llama.cpp.zmake`'s `common_sources`: upstream's `llama-common`.
const common_sources = [_][]const u8{
    "common/arg.cpp",                      "common/chat-auto-parser-generator.cpp",
    "common/chat-auto-parser-helpers.cpp", "common/chat-diff-analyzer.cpp",
    "common/chat-peg-parser.cpp",          "common/chat.cpp",
    "common/common.cpp",                   "common/console.cpp",
    "common/debug.cpp",                    "common/download.cpp",
    "common/fit.cpp",                      "common/file-lock.cpp",
    "common/hf-cache.cpp",                 "common/imatrix-loader.cpp",
    "common/json-schema-to-grammar.cpp",   "common/json-schema.cpp",
    "common/json.cpp",                     "common/llguidance.cpp",
    "common/log.cpp",                      "common/ngram-cache.cpp",
    "common/ngram-map.cpp",                "common/ngram-mod.cpp",
    "common/peg-parser.cpp",               "common/preset.cpp",
    "common/reasoning-budget.cpp",         "common/sampling.cpp",
    "common/speculative.cpp",              "common/subproc.cpp",
    "common/trie.cpp",                     "common/unicode.cpp",
    "common/jinja/lexer.cpp",              "common/jinja/parser.cpp",
    "common/jinja/runtime.cpp",            "common/jinja/value.cpp",
    "common/jinja/string.cpp",             "common/jinja/caps.cpp",
    // common/parsers/sources.cmake's LLAMA_CHAT_PARSERS_SOURCES
    "common/parsers/parsers.cpp",          "common/parsers/cohere2moe.cpp",
    "common/parsers/deepseek.cpp",         "common/parsers/functionary-v3-2.cpp",
    "common/parsers/gemma4.cpp",           "common/parsers/gigachat-v3.cpp",
    "common/parsers/gpt-oss.cpp",          "common/parsers/harmony-xml.cpp",
    "common/parsers/k2-horizon.cpp",       "common/parsers/kimi-k2.cpp",
    "common/parsers/kimi-k3.cpp",          "common/parsers/ling3.cpp",
    "common/parsers/lfm2.cpp",             "common/parsers/llm-jp-harmony.cpp",
    "common/parsers/minicpm5.cpp",         "common/parsers/minimax-m3.cpp",
    "common/parsers/ministral3.cpp",       "common/parsers/muse-glimmer.cpp",
    "common/parsers/muse-spark.cpp",       "common/parsers/qwen3-coder.cpp",
    "common/parsers/translate-gemma.cpp",
};

/// `llama.cpp.zmake`'s `server_context_sources`, `server_impl_sources` and
/// the executable's `main.cpp`: upstream's `llama-server`.
const server_sources = [_][]const u8{
    "tools/server/server-chat.cpp",     "tools/server/server-task.cpp",
    "tools/server/server-queue.cpp",    "tools/server/server-common.cpp",
    "tools/server/server-context.cpp",  "tools/server/server-realtime.cpp",
    "tools/server/server-stream.cpp",   "tools/server/server-tools.cpp",
    "tools/server/server-mcp.cpp",      "tools/server/server-schema.cpp",
    "tools/server/server-decision.cpp", "tools/server/server.cpp",
    "tools/server/server-connect.cpp",  "tools/server/server-http.cpp",
    "tools/server/server-models.cpp",   "tools/server/server-models-sched.cpp",
    "tools/server/main.cpp",
};

/// `llama.cpp.zmake`'s `httplib_defines`.
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
    config: *std.Build.Step.Options,
    reference: llamacpp.Artifacts,
    /// The CUDA toolkit when ggml has a CUDA backend: upstream's CMake makes
    /// `GGML_USE_CUDA` a public definition of `ggml`, so its tools compile
    /// with it, and `pca.hpp` picks its backend on it. Upstream's tools link
    /// its libraries through `llamacpp.addCudaLibraries`, as every
    /// executable over a CUDA `libggml.a` must.
    cuda: ?llamacpp.Cuda = null,
    /// Upstream's libmtmd (`addMtmdRef`), built once and shared with
    /// `build/mtmd.zig`.
    mtmd_ref: *std.Build.Step.Compile,
    /// `zigjinja`, the CLI's one dependency, for `llama-server`'s
    /// `cli/common_chat/` module: upstream's server parses the chat
    /// template with its Jinja engine at startup, `--no-jinja` or not, for
    /// the capabilities `/props` reports.
    jinja: ?*std.Build.Module = null,
    /// The web UI both servers embed (`build/ui.zig`).
    ui: @import("ui.zig").Ui,
};

/// The target triple `--version` prints, as `llama.cpp.zmake` formats it.
fn buildTarget(b: *std.Build, target: std.Build.ResolvedTarget) []const u8 {
    return b.fmt("{s}-{s}", .{ @tagName(target.result.cpu.arch), @tagName(target.result.os.tag) });
}

/// Adds `tools` (our ports, into `zig-out/bin`) and `tools-ref` (upstream's,
/// into `zig-out/tools-ref/bin`).
pub fn add(b: *std.Build, opts: Options) !void {
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
    build_info.addOption(u32, "build_number", build_number);
    build_info.addOption([]const u8, "commit", commit);
    build_info.addOption([]const u8, "compiler", "Zig " ++ builtin.zig_version_string ++ " clang");
    build_info.addOption([]const u8, "build_target", buildTarget(b, opts.target));

    // `cli/fit.zig`, the port of `common/fit.cpp`, shared with the CLI.
    const fit_module = b.createModule(.{
        .root_source_file = b.path("cli/fit.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
    });
    fit_module.addImport("llama", opts.reference.llama.root_module);

    const common_module = b.createModule(.{
        .root_source_file = b.path("tools/common/module.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
    });
    common_module.addImport("tools_h", tools_h_module);
    common_module.addOptions("build_info", build_info);
    common_module.addImport("cli_fit", fit_module);
    // libllama's module, for what `common/speculative.cpp` reaches through
    // `llama-ext.h` (C++ linkage, so not in `tools_h`) and for `libcxx`.
    common_module.addImport("llama", opts.reference.llama.root_module);

    const tools_step = b.step("tools", "Build the ported upstream tools");
    // libmtmd's port, for the mtmd tools: its `exports.zig` gives them the
    // `mtmd.h` and `mtmd-helper.h` symbols they call through `tools_h`.
    const mtmd_module = mtmd_build.module(b, .{
        .target = opts.target,
        .optimize = opts.optimize,
        .sdk = opts.sdk,
        .reference = opts.reference,
    });
    // Stage 9's modules, for the server's data layer (`tools/server/module.zig`)
    // and `llama-mtmd-cli`'s chat templates (`common_chat_templates_init`):
    // `json` is `cli_fit`'s `cli/json/` under the name `cli/common_chat/`
    // imports (`tools/server/json_module.zig` says why).
    const json_module = b.createModule(.{ .root_source_file = b.path("tools/server/json_module.zig"), .target = opts.target, .optimize = opts.optimize, .link_libc = true });
    json_module.addImport("cli_fit", fit_module);
    const peg_module = b.createModule(.{ .root_source_file = b.path("cli/peg/module.zig"), .target = opts.target, .optimize = opts.optimize, .link_libc = true });
    peg_module.addImport("json", json_module);
    peg_module.addImport("llama", opts.reference.llama.root_module);
    const chat_module = b.createModule(.{ .root_source_file = b.path("cli/common_chat/module.zig"), .target = opts.target, .optimize = opts.optimize, .link_libc = true });
    chat_module.addImport("json", json_module);
    chat_module.addImport("peg", peg_module);
    chat_module.addImport("llama", opts.reference.llama.root_module);
    if (opts.jinja) |j| chat_module.addImport("zigjinja", j);
    for (tools ++ mtmd_tools, 0..) |t, ti| {
        const mod = b.createModule(.{
            .root_source_file = b.path(t.port),
            .target = opts.target,
            .optimize = opts.optimize,
            .link_libc = true,
            .link_libcpp = true,
        });
        mod.addImport("tools_common", common_module);
        mod.addImport("tools_h", tools_h_module);
        mod.addImport("cli_fit", fit_module);
        mod.addOptions("config", opts.config);
        const backends = b.addOptions();
        backends.addOption(bool, "cuda", opts.cuda != null);
        mod.addOptions("backends", backends);
        // libllama's module, compiled in as the CLI does: referencing it
        // exports `llama.h`, and it links `libggml` itself.
        mod.addImport("llama", opts.reference.llama.root_module);
        if (ti >= tools.len) {
            mod.addImport("mtmd", mtmd_module);
            mod.addImport("common_chat", chat_module);
        }
        llamacpp.addSdkPaths(b, mod, opts.sdk);
        llamacpp.addLibgcc(b, mod);
        const exe = b.addExecutable(.{ .name = b.fmt("llama-{s}", .{t.name}), .root_module = mod });
        tools_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    }
    // Our `llama-server` (Stage 10, Decision 65), over the same modules as
    // the mtmd tools: it takes images and audio once step 8 lands. Not built
    // for Windows: its port there is out of scope (Marc, 2026-11-04).
    if (opts.target.result.os.tag != .windows) {
        const mod = b.createModule(.{
            .root_source_file = b.path("tools/server/main.zig"),
            .target = opts.target,
            .optimize = opts.optimize,
            .link_libc = true,
            .link_libcpp = true,
        });
        mod.addImport("tools_common", common_module);
        mod.addImport("tools_h", tools_h_module);
        mod.addImport("cli_fit", fit_module);
        mod.addImport("llama", opts.reference.llama.root_module);
        mod.addImport("mtmd", mtmd_module);
        mod.addImport("ui_assets", opts.ui.port_module);
        mod.addImport("json", json_module);
        mod.addImport("common_chat", chat_module);
        mod.addImport("peg", peg_module);
        llamacpp.addSdkPaths(b, mod, opts.sdk);
        llamacpp.addLibgcc(b, mod);
        const exe = b.addExecutable(.{ .name = "llama-server", .root_module = mod });
        tools_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
        // The router, the downloader and --connect, which no other step's
        // test root reaches: `tools/server/main.zig`'s test block names them.
        const host_test = b.addTest(.{ .root_module = mod });
        const host_test_step = b.step("test-server-host", "Run the unit tests of the router, the downloader and --connect (tools/server/models*.zig, download.zig, connect.zig)");
        host_test_step.dependOn(&b.addRunArtifact(host_test).step);
    }

    const ref_step = b.step("tools-ref", "Build upstream's tools over libllama-ref, the tool gates' references");
    const common_ref = addCommonRef(b, opts);
    // Installed for the in-process gates over upstream's `common/`
    // (`scripts/chat-msg-diff`), which link it beside a harness of their own.
    const common_ref_install = b.addInstallArtifact(common_ref, .{});
    ref_step.dependOn(&common_ref_install.step);
    // That library alone: on Windows `tools-ref` cannot build (its server and
    // multimodal tools have no Windows build yet), and peg-diff and
    // schema-diff need nothing else from it.
    const common_ref_step = b.step("common-ref", "Build upstream's libllama-common over libllama-ref alone");
    common_ref_step.dependOn(&common_ref_install.step);
    // Upstream's mtmd, and the two tools over it: the multimodal CLI and TTS.
    const mtmd_ref = opts.mtmd_ref;
    const mtmd_ref_step = b.step("mtmd-ref", "Build upstream's libmtmd over libllama-ref, the mtmd gates' reference");
    mtmd_ref_step.dependOn(&b.addInstallArtifact(mtmd_ref, .{}).step);
    for (mtmd_tools) |t| {
        const mod = refModule(b, opts);
        mod.addCSourceFiles(.{ .root = b.path(root), .files = t.ref, .flags = &cxx_flags });
        mod.linkLibrary(common_ref);
        mod.linkLibrary(mtmd_ref);
        mod.linkLibrary(opts.reference.llama_ref);
        mod.linkLibrary(opts.reference.ggml);
        llamacpp.addSdkPaths(b, mod, opts.sdk);
        llamacpp.addLibgcc(b, mod);
        llamacpp.addCudaRuntime(b, mod, opts.cuda);
        const exe = b.addExecutable(.{ .name = b.fmt("llama-{s}", .{t.name}), .root_module = mod });
        const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "tools-ref/bin" } } });
        ref_step.dependOn(&install.step);
        mtmd_ref_step.dependOn(&install.step);
    }
    // Upstream's `llama-server` (Stage 10, Decision 65), as `llama.cpp.zmake`
    // builds it: `server-context` and `llama-server-impl` over common, mtmd
    // and cpp-httplib, with the web UI embedded under -Dui (zmake's stub,
    // and no assets, without it).
    {
        const mod = refModule(b, opts);
        mod.addIncludePath(b.path(root ++ "/tools/server"));
        mod.addIncludePath(b.path(root ++ "/vendor/cpp-httplib"));
        mod.addIncludePath(opts.ui.ref_include);
        const flags = if (opts.cuda != null) &(cxx_flags ++ .{"-DGGML_USE_CUDA"}) else &cxx_flags;
        mod.addCSourceFiles(.{ .root = b.path(root), .files = &server_sources, .flags = flags });
        mod.addCSourceFile(.{ .file = opts.ui.ref_cpp, .flags = &cxx_flags });
        mod.linkLibrary(common_ref);
        mod.linkLibrary(mtmd_ref);
        mod.linkLibrary(opts.reference.llama_ref);
        mod.linkLibrary(opts.reference.ggml);
        llamacpp.addSdkPaths(b, mod, opts.sdk);
        llamacpp.addLibgcc(b, mod);
        llamacpp.addCudaRuntime(b, mod, opts.cuda);
        const exe = b.addExecutable(.{ .name = "llama-server", .root_module = mod });
        const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "tools-ref/bin" } } });
        ref_step.dependOn(&install.step);
        const server_ref_step = b.step("server-ref", "Build upstream's llama-server over libllama-ref, the server gates' reference");
        server_ref_step.dependOn(&install.step);
    }
    // `scripts/spec-diff`'s two sides: `harness/spec_diff.zig` over the port's
    // `tools/common/`, `harness/spec_ref.cpp` over upstream's `common/`.
    {
        const spec_step = b.step("spec-diff", "Build the spec-diff harnesses: the port's common/ sampling and speculative decoding, and upstream's");
        const port = b.createModule(.{
            .root_source_file = b.path("harness/spec_diff.zig"),
            .target = opts.target,
            .optimize = opts.optimize,
            .link_libc = true,
            .link_libcpp = true,
        });
        port.addImport("tools_common", common_module);
        port.addImport("tools_h", tools_h_module);
        port.addImport("cli_fit", fit_module);
        port.addImport("llama", opts.reference.llama.root_module);
        llamacpp.addSdkPaths(b, port, opts.sdk);
        llamacpp.addLibgcc(b, port);
        const port_exe = b.addExecutable(.{ .name = "spec_diff", .root_module = port });
        spec_step.dependOn(&b.addInstallArtifact(port_exe, .{ .dest_dir = .{ .override = .{ .custom = "spec-diff" } } }).step);

        const ref = refModule(b, opts);
        ref.addCSourceFiles(.{ .root = b.path("harness"), .files = &.{"spec_ref.cpp"}, .flags = &cxx_flags });
        ref.linkLibrary(common_ref);
        ref.linkLibrary(opts.reference.llama_ref);
        ref.linkLibrary(opts.reference.ggml);
        llamacpp.addSdkPaths(b, ref, opts.sdk);
        llamacpp.addLibgcc(b, ref);
        llamacpp.addCudaRuntime(b, ref, opts.cuda);
        const ref_exe = b.addExecutable(.{ .name = "spec_ref", .root_module = ref });
        spec_step.dependOn(&b.addInstallArtifact(ref_exe, .{ .dest_dir = .{ .override = .{ .custom = "spec-diff" } } }).step);

        // gives the draft fixtures a vocabulary, so DFlash has a mask token
        const fx = b.createModule(.{ .target = opts.target, .optimize = opts.optimize, .link_libc = true, .link_libcpp = true });
        fx.addCSourceFiles(.{ .root = b.path("harness"), .files = &.{"spec_fixtures.c"}, .flags = &.{"-std=c11"} });
        fx.addIncludePath(b.path(root ++ "/ggml/include"));
        fx.linkLibrary(opts.reference.ggml);
        llamacpp.addSdkPaths(b, fx, opts.sdk);
        llamacpp.addLibgcc(b, fx);
        // libggml links the CUDA libraries by name on a CUDA build; this
        // says where they are, as for every other reference here
        llamacpp.addCudaRuntime(b, fx, opts.cuda);
        const fx_exe = b.addExecutable(.{ .name = "spec_fixtures", .root_module = fx });
        spec_step.dependOn(&b.addInstallArtifact(fx_exe, .{ .dest_dir = .{ .override = .{ .custom = "spec-diff" } } }).step);
    }
    for (tools) |t| {
        const mod = refModule(b, opts);
        mod.addCSourceFiles(.{ .root = b.path(root), .files = t.ref, .flags = if (opts.cuda != null) &(cxx_flags ++ .{"-DGGML_USE_CUDA"}) else &cxx_flags });
        if (opts.cuda != null) mod.addIncludePath(b.path("llama.cpp/ggml/include"));
        mod.linkLibrary(common_ref);
        mod.linkLibrary(opts.reference.llama_ref);
        mod.linkLibrary(opts.reference.ggml);
        llamacpp.addSdkPaths(b, mod, opts.sdk);
        llamacpp.addLibgcc(b, mod);
        llamacpp.addCudaRuntime(b, mod, opts.cuda);
        const exe = b.addExecutable(.{ .name = b.fmt("llama-{s}", .{t.name}), .root_module = mod });
        ref_step.dependOn(&b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "tools-ref/bin" } } }).step);
    }
}

fn refModule(b: *std.Build, opts: anytype) *std.Build.Module {
    const mod = b.createModule(.{
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        // Reference code compiles as shipped; see `addLlamaRef`.
        .sanitize_c = .off,
    });
    mod.addIncludePath(b.path(root));
    mod.addIncludePath(b.path(root ++ "/include"));
    // `src` stays out: its `unicode.h` shadows `common/unicode.h`.
    mod.addIncludePath(b.path(root ++ "/common"));
    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/src"));
    mod.addIncludePath(b.path(root ++ "/vendor"));
    mod.addIncludePath(b.path(root ++ "/tools/mtmd"));
    // `LLAMA_SUBPROCESS`, a PUBLIC definition of `llama-common` on every
    // platform but iOS, Android and WASM (CMakeLists.txt:104-110,
    // common/CMakeLists.txt:142), as `llama.cpp.zmake`'s `subprocess_defines`
    // gives it: without it upstream's router mode refuses to start ("subprocess
    // is not enabled on this build") and `common_subproc::create` always fails.
    if (opts.target.result.os.tag != .ios) mod.addCMacro("LLAMA_SUBPROCESS", "");
    return mod;
}

/// Upstream's `llama-common` with its `build-info.cpp` and `cpp-httplib`, as
/// `llama.cpp.zmake` builds them.
fn addCommonRef(b: *std.Build, opts: Options) *std.Build.Step.Compile {
    // Configured from upstream's build-info.cpp.in, as common/CMakeLists.txt
    // does, so a change to the template cannot leave this behind.
    const build_info = b.addConfigHeader(.{
        .style = .{ .cmake = b.path(root ++ "/common/build-info.cpp.in") },
        .include_path = "build-info.cpp",
    }, .{
        .LLAMA_BUILD_NUMBER = build_number,
        .LLAMA_BUILD_COMMIT = commit,
        .BUILD_COMPILER = "Zig " ++ builtin.zig_version_string ++ " clang",
        .BUILD_TARGET = buildTarget(b, opts.target),
    }).getOutputFile();

    const mod = refModule(b, opts);
    mod.addCSourceFile(.{ .file = build_info, .flags = &cxx_flags });
    mod.addCSourceFiles(.{ .root = b.path(root), .files = &common_sources, .flags = &cxx_flags });
    mod.addIncludePath(b.path(root ++ "/vendor/cpp-httplib"));
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &.{"vendor/cpp-httplib/httplib.cpp"},
        .flags = &(cxx_flags ++ httplib_defines),
    });
    if (opts.target.result.os.tag == .windows) mod.linkSystemLibrary("ws2_32", .{});
    return b.addLibrary(.{ .name = "llama-common-ref", .root_module = mod, .linkage = .static });
}

/// Upstream's `mtmd` with `vendor::hash`, as `llama.cpp.zmake` builds them
/// (its `addCli`): stb_image and miniaudio are header-only and compiled into
/// `mtmd-helper.cpp`; no `MTMD_VIDEO`, as zmake defines none.
pub fn addMtmdRef(b: *std.Build, opts: anytype) *std.Build.Step.Compile {
    const mod = refModule(b, opts);
    mod.addIncludePath(b.path(root ++ "/vendor/hash"));
    mod.addCSourceFiles(.{ .root = b.path(root), .files = &mtmd_sources, .flags = &cxx_flags });
    var models: std.ArrayList([]const u8) = .empty;
    var dir = std.Io.Dir.cwd().openDir(b.graph.io, root ++ "/tools/mtmd/models", .{ .iterate = true }) catch @panic("cannot open tools/mtmd/models");
    defer dir.close(b.graph.io);
    var it = dir.iterate();
    while (it.next(b.graph.io) catch null) |e| {
        if (std.mem.endsWith(u8, e.name, ".cpp")) models.append(b.allocator, b.fmt("tools/mtmd/models/{s}", .{e.name})) catch @panic("OOM");
    }
    std.mem.sort([]const u8, models.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
    mod.addCSourceFiles(.{ .root = b.path(root), .files = models.items, .flags = &cxx_flags });
    // vendor::hash (vendor/hash/CMakeLists.txt): sha1.c is C++ despite its
    // extension, as zmake records.
    mod.addCSourceFiles(.{ .root = b.path(root), .files = &.{"vendor/hash/hash.cpp"}, .flags = &cxx_flags });
    mod.addCSourceFiles(.{ .root = b.path(root), .files = &.{"vendor/hash/sha1/sha1.c"}, .flags = &cxx_flags, .language = .cpp });
    mod.addCSourceFiles(.{ .root = b.path(root), .files = &.{ "vendor/hash/xxhash/xxhash.c", "vendor/hash/sha256/sha256.c" }, .flags = &.{ "-std=c11", "-w" } });
    llamacpp.addSdkPaths(b, mod, opts.sdk);
    return b.addLibrary(.{ .name = "mtmd-ref", .root_module = mod, .linkage = .static });
}
