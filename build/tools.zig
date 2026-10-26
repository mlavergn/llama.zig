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

const root = "llama.cpp";

/// `llama.cpp.zmake`'s build number and commit (its `build.zig`), so a tool's
/// `--version` reads the same from both sides of a gate.
pub const build_number = 10621;
pub const commit = "c1d0e7a00";

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

/// `llama.cpp.zmake`'s `common_sources`: upstream's `llama-common`.
const common_sources = [_][]const u8{
    "common/arg.cpp",                      "common/chat-auto-parser-generator.cpp",
    "common/chat-auto-parser-helpers.cpp", "common/chat-diff-analyzer.cpp",
    "common/chat-peg-parser.cpp",          "common/chat.cpp",
    "common/common.cpp",                   "common/console.cpp",
    "common/debug.cpp",                    "common/download.cpp",
    "common/fit.cpp",                      "common/hf-cache.cpp",
    "common/imatrix-loader.cpp",           "common/json-schema-to-grammar.cpp",
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
    /// Whether ggml has a CUDA backend: upstream's CMake makes
    /// `GGML_USE_CUDA` a public definition of `ggml`, so its tools compile
    /// with it, and `pca.hpp` picks its backend on it.
    cuda: bool = false,
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

    const tools_step = b.step("tools", "Build the ported upstream tools");
    for (tools) |t| {
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
        backends.addOption(bool, "cuda", opts.cuda);
        mod.addOptions("backends", backends);
        // libllama's module, compiled in as the CLI does: referencing it
        // exports `llama.h`, and it links `libggml` itself.
        mod.addImport("llama", opts.reference.llama.root_module);
        llamacpp.addSdkPaths(b, mod, opts.sdk);
        const exe = b.addExecutable(.{ .name = b.fmt("llama-{s}", .{t.name}), .root_module = mod });
        tools_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

    const ref_step = b.step("tools-ref", "Build upstream's tools over libllama-ref, the tool gates' references");
    const common_ref = addCommonRef(b, opts);
    for (tools) |t| {
        const mod = refModule(b, opts);
        mod.addCSourceFiles(.{ .root = b.path(root), .files = t.ref, .flags = if (opts.cuda) &(cxx_flags ++ .{"-DGGML_USE_CUDA"}) else &cxx_flags });
        if (opts.cuda) mod.addIncludePath(b.path("llama.cpp/ggml/include"));
        mod.linkLibrary(common_ref);
        mod.linkLibrary(opts.reference.llama_ref);
        mod.linkLibrary(opts.reference.ggml);
        llamacpp.addSdkPaths(b, mod, opts.sdk);
        const exe = b.addExecutable(.{ .name = b.fmt("llama-{s}", .{t.name}), .root_module = mod });
        ref_step.dependOn(&b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "tools-ref/bin" } } }).step);
    }
}

fn refModule(b: *std.Build, opts: Options) *std.Build.Module {
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
    return mod;
}

/// Upstream's `llama-common` with its `build-info.cpp` and `cpp-httplib`, as
/// `llama.cpp.zmake` builds them.
fn addCommonRef(b: *std.Build, opts: Options) *std.Build.Step.Compile {
    const build_info = b.addWriteFiles().add("build-info.cpp", b.fmt(
        \\#include "build-info.h"
        \\
        \\#include <cstdio>
        \\#include <string>
        \\
        \\int LLAMA_BUILD_NUMBER = {d};
        \\char const * LLAMA_COMMIT = "{s}";
        \\char const * LLAMA_COMPILER = "Zig {s} clang";
        \\char const * LLAMA_BUILD_TARGET = "{s}";
        \\
        \\int llama_build_number(void) {{ return LLAMA_BUILD_NUMBER; }}
        \\const char * llama_commit(void) {{ return LLAMA_COMMIT; }}
        \\const char * llama_compiler(void) {{ return LLAMA_COMPILER; }}
        \\const char * llama_build_target(void) {{ return LLAMA_BUILD_TARGET; }}
        \\
        \\const char * llama_build_info(void) {{
        \\    static std::string s = "b" + std::to_string(LLAMA_BUILD_NUMBER) + "-" + LLAMA_COMMIT;
        \\    return s.c_str();
        \\}}
        \\
        \\void llama_print_build_info(const char * llama_version) {{
        \\    fprintf(stderr, "version: %s (build %d, commit %s)\n", llama_version, llama_build_number(), llama_commit());
        \\    fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
        \\}}
        \\
    , .{ build_number, commit, builtin.zig_version_string, buildTarget(b, opts.target) }));

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
