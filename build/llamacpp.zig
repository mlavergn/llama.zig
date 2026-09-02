//! Build graph for the vendored llama.cpp reference tree.
//!
//! Replaces the CMake files under `llama.cpp/` for the macOS arm64
//! configuration this project targets. The sources are compiled exactly as
//! upstream ships them — nothing under `llama.cpp/` is modified — so this file
//! is a translation of the build description, not of the code.
//!
//! The `-D` options that `make buildmacos` passes to CMake are hardcoded here:
//! static libraries, Metal on with the shader library embedded, Accelerate on,
//! BLAS and OpenMP off, no curl.

const std = @import("std");

/// Where the reference tree lives, relative to the build root.
pub const root = "llama.cpp";

/// Warnings that fire on upstream's sources under Zig's clang and bundled
/// libc++ but say nothing about this project. Silenced so real diagnostics stay
/// visible rather than being buried.
const quiet = [_][]const u8{
    "-Wno-nullability-completeness",
    "-Wno-deprecated-declarations",
    "-Wno-unused-function",
    "-Wno-unused-variable",
};

const c_flags = [_][]const u8{"-std=c11"} ++ quiet;
const cxx_flags = [_][]const u8{"-std=c++17"} ++ quiet;
// Upstream's Metal sources use manual reference counting and bridge freely
// between void* and Obj-C object pointers, so ARC must stay off.
const objc_flags = [_][]const u8{"-std=c11"} ++ quiet;

/// The Metal kernels embedded into the binary, one shader library per entry.
///
/// Mirrors the `GGML_METAL_LIBS` X-macro in `ggml-metal-device.m`: each name
/// becomes a `ggml_metallib_<name>_{start,end}` symbol pair that the Obj-C
/// device layer reads at runtime.
const metal_kernels = [_][]const u8{
    "fa",   "mul_mv",  "mul_mm",          "quantize",  "softmax",
    "norm", "unary",   "binbcast",        "reduce",    "tri",
    "ssm",  "wkv",     "gated_delta_net", "solve_tri", "rope",
    "conv", "upscale", "argsort",         "pool",      "misc",
};

const ggml_base_sources = [_][]const u8{
    "ggml/src/ggml.c",
    "ggml/src/ggml-alloc.c",
    "ggml/src/ggml-quants.c",
};

const ggml_base_cxx_sources = [_][]const u8{
    "ggml/src/ggml.cpp",
    "ggml/src/ggml-backend.cpp",
    "ggml/src/ggml-backend-meta.cpp",
    "ggml/src/ggml-opt.cpp",
    "ggml/src/ggml-threading.cpp",
    "ggml/src/gguf.cpp",
    "ggml/src/ggml-backend-dl.cpp",
    "ggml/src/ggml-backend-reg.cpp",
};

const ggml_cpu_c_sources = [_][]const u8{
    "ggml/src/ggml-cpu/ggml-cpu.c",
    "ggml/src/ggml-cpu/quants.c",
    "ggml/src/ggml-cpu/arch/arm/quants.c",
};

const ggml_cpu_cxx_sources = [_][]const u8{
    "ggml/src/ggml-cpu/ggml-cpu.cpp",
    "ggml/src/ggml-cpu/repack.cpp",
    "ggml/src/ggml-cpu/hbm.cpp",
    "ggml/src/ggml-cpu/traits.cpp",
    "ggml/src/ggml-cpu/binary-ops.cpp",
    "ggml/src/ggml-cpu/unary-ops.cpp",
    "ggml/src/ggml-cpu/vec.cpp",
    "ggml/src/ggml-cpu/ops.cpp",
    "ggml/src/ggml-cpu/amx/amx.cpp",
    "ggml/src/ggml-cpu/amx/mmq.cpp",
    "ggml/src/ggml-cpu/llamafile/sgemm.cpp",
    "ggml/src/ggml-cpu/arch/arm/repack.cpp",
};

const ggml_metal_cxx_sources = [_][]const u8{
    "ggml/src/ggml-metal/ggml-metal.cpp",
    "ggml/src/ggml-metal/ggml-metal-device.cpp",
    "ggml/src/ggml-metal/ggml-metal-common.cpp",
    "ggml/src/ggml-metal/ggml-metal-ops.cpp",
    "ggml/src/ggml-metal/ggml-metal-tuning.cpp",
};

const ggml_metal_objc_sources = [_][]const u8{
    "ggml/src/ggml-metal/ggml-metal-device.m",
    "ggml/src/ggml-metal/ggml-metal-context.m",
};

const llama_sources = [_][]const u8{
    "src/llama.cpp",
    "src/llama-adapter.cpp",
    "src/llama-arch.cpp",
    "src/llama-batch.cpp",
    "src/llama-chat.cpp",
    "src/llama-context.cpp",
    "src/llama-cparams.cpp",
    "src/llama-grammar.cpp",
    "src/llama-graph.cpp",
    "src/llama-hparams.cpp",
    "src/llama-impl.cpp",
    "src/llama-io.cpp",
    "src/llama-kv-cache.cpp",
    "src/llama-kv-cache-iswa.cpp",
    "src/llama-kv-cache-dsa.cpp",
    "src/llama-kv-cache-dsa-iswa.cpp",
    "src/llama-kv-cache-msa.cpp",
    "src/llama-kv-cache-dsv4.cpp",
    "src/llama-memory.cpp",
    "src/llama-memory-hybrid.cpp",
    "src/llama-memory-hybrid-iswa.cpp",
    "src/llama-memory-recurrent.cpp",
    "src/llama-mmap.cpp",
    "src/llama-model-loader.cpp",
    "src/llama-model-saver.cpp",
    "src/llama-model.cpp",
    "src/llama-quant.cpp",
    "src/llama-sampler.cpp",
    "src/llama-vocab.cpp",
    "src/unicode-data.cpp",
    "src/unicode.cpp",
};

/// Everything the reference build produces, so callers can wire up whichever
/// pieces they need without reaching back into this file's internals.
pub const Artifacts = struct {
    /// ggml plus its CPU and Metal backends, as one static archive.
    ggml: *std.Build.Step.Compile,
    /// libllama, linked against `ggml`.
    llama: *std.Build.Step.Compile,
};

/// Options that shape the reference build.
pub const Options = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    /// macOS SDK path, used as the framework search path. Null off macOS.
    sdk: ?[]const u8 = null,
};

/// Declares the ggml and llama static libraries and returns them.
///
/// Compiles the vendored sources with the macOS arm64 flag set. Must be called
/// once per build invocation; the returned artifacts are owned by `b`.
///
/// Parameters:
/// - `b`: the build graph the compile steps are registered on.
/// - `opts`: target, optimize mode, and macOS SDK path.
///
/// Return: the built libraries; propagates errors from reading `src/models/`.
pub fn add(b: *std.Build, opts: Options) !Artifacts {
    const ggml = try addGgml(b, opts);
    const llama = try addLlama(b, opts, ggml);
    return .{ .ggml = ggml, .llama = llama };
}

/// Builds ggml with its CPU and Metal backends as a single static archive.
///
/// Upstream splits these into separate CMake targets so that backends can be
/// loaded dynamically. With static linking and `GGML_BACKEND_DL` off, the split
/// carries no meaning: `ggml-backend-reg.cpp` calls each backend's registration
/// function directly under an `#ifdef`, so one archive links identically.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target, optimize mode, and macOS SDK path.
///
/// Return: the ggml static library; propagates allocation failure.
fn addGgml(b: *std.Build, opts: Options) !*std.Build.Step.Compile {
    const mod = b.createModule(.{
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        // Upstream is not UBSan-clean -- it relies on pointer arithmetic that
        // is technically undefined but universally works, and Zig enables the
        // C sanitizers in Debug by default. This is reference code we compile
        // as shipped, not code we fix, so the checks come off.
        .sanitize_c = .off,
    });

    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/src"));
    mod.addIncludePath(b.path(root ++ "/ggml/src/ggml-cpu"));
    mod.addIncludePath(b.path(root ++ "/ggml/src/ggml-metal"));

    // Backend selection. Upstream sets these from the -D options; this build
    // targets one configuration, so they are fixed.
    const defines = [_][]const u8{
        "-DGGML_USE_CPU",
        "-DGGML_USE_METAL",
        "-DGGML_SCHED_MAX_COPIES=4",
        "-DGGML_USE_ACCELERATE",
        "-DACCELERATE_NEW_LAPACK",
        "-DACCELERATE_LAPACK_ILP64",
        "-DGGML_USE_LLAMAFILE",
        "-DGGML_METAL_EMBED_LIBRARY",
        "-DGGML_VERSION=\"0.3.0\"",
        "-DGGML_COMMIT=\"c1d0e7a00\"",
    };

    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ggml_base_sources ++ &ggml_cpu_c_sources,
        .flags = &(c_flags ++ defines),
    });
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ggml_base_cxx_sources ++ &ggml_cpu_cxx_sources ++ &ggml_metal_cxx_sources,
        .flags = &(cxx_flags ++ defines),
    });
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ggml_metal_objc_sources,
        .flags = &(objc_flags ++ defines),
    });

    for (metal_kernels) |kind| {
        mod.addAssemblyFile(embedMetalKernel(b, kind));
    }

    if (opts.sdk) |sdk| mod.addFrameworkPath(.{ .cwd_relative = sdk });
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});
    mod.linkFramework("MetalKit", .{});
    mod.linkFramework("Accelerate", .{});

    return b.addLibrary(.{ .name = "ggml", .root_module = mod, .linkage = .static });
}

/// Flattens one Metal kernel and returns the assembly stub that embeds it.
///
/// Runs `build/metal_embed.zig` as a build-time tool, which stands in for the
/// `cat`/`sed` pipeline upstream's CMake uses. The stub `.incbin`s the
/// flattened shader source into a `__DATA,__ggml_metallib` section; the Metal
/// driver compiles it at load time, so no `xcrun metal` step is involved.
///
/// Parameters:
/// - `b`: the build graph the run step is registered on.
/// - `kind`: kernel name, matching a `kernels/<kind>.metal` source.
///
/// Return: the generated `.s` file, valid for the build's lifetime.
fn embedMetalKernel(b: *std.Build, kind: []const u8) std.Build.LazyPath {
    const tool = b.addExecutable(.{
        .name = "metal_embed",
        .root_module = b.createModule(.{
            .root_source_file = b.path("build/metal_embed.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });

    const run = b.addRunArtifact(tool);
    run.addArg(kind);
    const out_metal = run.addOutputFileArg(b.fmt("ggml-metal-embed-{s}.metal", .{kind}));
    const out_asm = run.addOutputFileArg(b.fmt("ggml-metal-embed-{s}.s", .{kind}));
    _ = out_metal;

    const metal = root ++ "/ggml/src/ggml-metal";
    run.addFileArg(b.path(metal ++ "/../ggml-common.h"));
    run.addFileArg(b.path(metal ++ "/ggml-metal-impl.h"));
    run.addFileArg(b.path(metal ++ "/kernels/common.h"));
    run.addFileArg(b.path(metal ++ "/kernels/dequantize.h"));
    run.addFileArg(b.path(metal ++ "/kernels/quantize.h"));
    run.addFileArg(b.path(b.fmt(metal ++ "/kernels/{s}.metal", .{kind})));

    return out_asm;
}

/// Builds libllama, including every architecture under `src/models/`.
///
/// Upstream globs `models/*.cpp`; this reads the directory for the same reason,
/// rather than carrying a 151-entry list that would go stale on the next sync.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target, optimize mode, and macOS SDK path.
/// - `ggml`: the ggml library to link against.
///
/// Return: the llama static library; propagates directory-read and allocation
/// failures.
fn addLlama(b: *std.Build, opts: Options, ggml: *std.Build.Step.Compile) !*std.Build.Step.Compile {
    const mod = b.createModule(.{
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        // Upstream is not UBSan-clean -- it relies on pointer arithmetic that
        // is technically undefined but universally works, and Zig enables the
        // C sanitizers in Debug by default. This is reference code we compile
        // as shipped, not code we fix, so the checks come off.
        .sanitize_c = .off,
    });

    mod.addIncludePath(b.path(root ++ "/include"));
    mod.addIncludePath(b.path(root ++ "/src"));
    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/src"));

    var files: std.ArrayList([]const u8) = .empty;
    try files.appendSlice(b.allocator, &llama_sources);

    var dir = try b.build_root.handle.openDir(b.graph.io, root ++ "/src/models", .{ .iterate = true });
    defer dir.close(b.graph.io);
    var it = dir.iterate();
    while (try it.next(b.graph.io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".cpp")) continue;
        try files.append(b.allocator, b.fmt("src/models/{s}", .{entry.name}));
    }

    const defines = [_][]const u8{
        "-DLLAMA_VERSION=\"0.3.0\"",
        "-DLLAMA_COMMIT=\"c1d0e7a00\"",
        "-DGGML_USE_CPU",
        "-DGGML_USE_METAL",
    };

    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = files.items,
        .flags = &(cxx_flags ++ defines),
    });

    if (opts.sdk) |sdk| mod.addFrameworkPath(.{ .cwd_relative = sdk });
    mod.linkLibrary(ggml);

    return b.addLibrary(.{ .name = "llama", .root_module = mod, .linkage = .static });
}
