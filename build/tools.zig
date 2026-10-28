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
    /// Upstream's libmtmd (`addMtmdRef`), built once and shared with
    /// `build/mtmd.zig`.
    mtmd_ref: *std.Build.Step.Compile,
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

    const tools_step = b.step("tools", "Build the ported upstream tools");
    // libmtmd's port, for the mtmd tools: its `exports.zig` gives them the
    // `mtmd.h` and `mtmd-helper.h` symbols they call through `tools_h`.
    const mtmd_module = mtmd_build.module(b, .{
        .target = opts.target,
        .optimize = opts.optimize,
        .sdk = opts.sdk,
        .reference = opts.reference,
    });
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
        backends.addOption(bool, "cuda", opts.cuda);
        mod.addOptions("backends", backends);
        // libllama's module, compiled in as the CLI does: referencing it
        // exports `llama.h`, and it links `libggml` itself.
        mod.addImport("llama", opts.reference.llama.root_module);
        if (ti >= tools.len) mod.addImport("mtmd", mtmd_module);
        llamacpp.addSdkPaths(b, mod, opts.sdk);
        const exe = b.addExecutable(.{ .name = b.fmt("llama-{s}", .{t.name}), .root_module = mod });
        tools_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

    const ref_step = b.step("tools-ref", "Build upstream's tools over libllama-ref, the tool gates' references");
    const common_ref = addCommonRef(b, opts);
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
        const exe = b.addExecutable(.{ .name = b.fmt("llama-{s}", .{t.name}), .root_module = mod });
        const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "tools-ref/bin" } } });
        ref_step.dependOn(&install.step);
        mtmd_ref_step.dependOn(&install.step);
    }
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
