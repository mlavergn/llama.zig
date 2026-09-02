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
    // Empty: ggml-alloc.c is ported (src/ggml/alloc.zig), ggml.c is ported
    // (src/ggml/*.zig), and ggml-quants.c is ported (src/ggml/quants/).
    // Nothing under ggml/src/ is compiled as C any more.
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
    // Empty: ggml-cpu.c, ggml-cpu/quants.c and ggml-cpu/arch/arm/quants.c are
    // all ported. **No C compiles anywhere under ggml/src/ any more.**
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
    /// The ggml module, exposed so the ported Zig under `src/ggml/` can be
    /// unit-tested. Tests need the same C sources and frameworks the library
    /// does, because ported code calls back into the parts still in C.
    ggml_module: *std.Build.Module,
};

/// Options that shape the reference build.
pub const Options = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    /// macOS SDK path, used as the framework search path. Null off macOS.
    sdk: ?[]const u8 = null,
    /// Makes ported code abort on a hot path, to prove it is executing.
    ///
    /// Comparing output cannot distinguish code that is correct from code that
    /// never runs. A swapped symbol can be defined, referenced and linked while
    /// the program reaches a different implementation entirely -- not
    /// hypothetical, that is what sank the incremental swap trial.
    /// `scripts/probe-ported` builds with this on and requires the abort, so a
    /// silent bypass fails loudly.
    probe_ported: bool = false,
    /// The same, for the ported CPU backend.
    ///
    /// A separate flag because a single one would only prove whichever site
    /// runs first. `ggml-alloc.c` runs while the graph is being planned and
    /// `ggml-cpu.c` while it is being computed, so each needs its own run to
    /// be shown reachable.
    probe_cpu: bool = false,
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
    return .{ .ggml = ggml, .llama = llama, .ggml_module = ggml.root_module };
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
        // Rooting the module at our Zig barrel is what lets ported code share
        // an archive with the C it replaces: Zig objects and C objects land in
        // the same `libggml.a`, and the C++ above links against whichever of
        // the two currently defines a symbol.
        .root_source_file = b.path("src/ggml/module.zig"),
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

    const options = b.addOptions();
    options.addOption(bool, "probe_ported", opts.probe_ported);
    options.addOption(bool, "probe_cpu", opts.probe_cpu);
    mod.addOptions("config", options);

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

    // ggml.c is ported and swapped out: see src/ggml/{impl,types,context,
    // runtime,ops,graph,quantize}.zig, reached through the module barrel this
    // library is rooted at. Nothing compiles it any more, so the linker's
    // missing-symbol errors are what prove the port is complete.
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

/// The C translation units the ported-but-not-yet-swapped Zig depends on.
///
/// Deliberately short: it must never include a file whose symbols overlap the
/// ported code, or the test link fails with duplicates. See
/// `src/ggml/ported.zig`.
const ported_test_c_sources = [_][]const u8{
    // Empty: every C translation unit under ggml/src/ is ported. The traits
    // table resolves entirely to Zig.
};

const ported_test_cxx_sources = [_][]const u8{
    "ggml/src/ggml-threading.cpp",
    // Linkable since the graph section landed: it needs the graph and
    // allocator symbols, which the port now provides. Before that,
    // `ported.zig` stubbed the one backend function the ported code called.
    "ggml/src/ggml-backend.cpp",
    // ggml-backend.cpp calls into the meta-buffer backend; it comes along.
    "ggml/src/ggml-backend-meta.cpp",
    // The ported `ggml-cpu.c` dispatches into these: the op kernels, the
    // vector helpers, the extra-buffer hooks, and the llamafile fast path.
    // Everything under ggml-cpu/ except ggml-cpu.c itself, which is ours.
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

/// Builds a static library from the ported Zig alone, for verification.
///
/// The point is to make the ported constructors *reachable* before `ggml.c` is
/// swapped in. Linking a driver against `libggml.a` proves nothing today: that
/// archive still contains `ggml.c`, so a caller reaches the C implementation
/// and a comparison would be C against C. This library contains only the Zig,
/// plus the two C translation units it genuinely depends on, so anything that
/// links it *must* reach the port.
///
/// Return: the library; callers link it instead of `libggml.a`.
pub fn addPortedGgml(b: *std.Build, opts: Options) *std.Build.Step.Compile {
    const mod = b.createModule(.{
        .root_source_file = b.path("src/ggml/ported.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        .sanitize_c = .off,
    });

    const options = b.addOptions();
    options.addOption(bool, "probe_ported", opts.probe_ported);
    options.addOption(bool, "probe_cpu", opts.probe_cpu);
    mod.addOptions("config", options);

    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/src"));
    // The CPU sources reach their own headers unqualified.
    mod.addIncludePath(b.path(root ++ "/ggml/src/ggml-cpu"));

    const defines = [_][]const u8{
        "-DGGML_USE_CPU",
        "-DGGML_SCHED_MAX_COPIES=4",
        "-DGGML_USE_ACCELERATE",
        "-DACCELERATE_NEW_LAPACK",
        "-DACCELERATE_LAPACK_ILP64",
        "-DGGML_USE_LLAMAFILE",
        "-DGGML_VERSION=\"0.3.0\"",
        "-DGGML_COMMIT=\"c1d0e7a00\"",
    };

    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ported_test_c_sources,
        .flags = &(c_flags ++ defines),
    });
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ported_test_cxx_sources,
        .flags = &(cxx_flags ++ defines),
    });

    if (opts.sdk) |sdk| mod.addFrameworkPath(.{ .cwd_relative = sdk });
    // `-DGGML_USE_ACCELERATE` sends the vector helpers in `ops.cpp` and
    // `binary-ops.cpp` to vDSP, so the framework has to come with them.
    mod.linkFramework("Accelerate", .{});
    return b.addLibrary(.{ .name = "ggml-ported", .root_module = mod, .linkage = .static });
}

/// Declares a test step for the ported Zig that has not been swapped in yet.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target, optimize mode, and macOS SDK path.
///
/// Return: the test module, for the caller to attach to a step.
pub fn portedTestModule(b: *std.Build, opts: Options) *std.Build.Module {
    return portedTestModuleInner(b, opts, "src/ggml/ported.zig", &ported_test_c_sources);
}

fn portedTestModuleInner(
    b: *std.Build,
    opts: Options,
    root_source: []const u8,
    c_sources: []const []const u8,
) *std.Build.Module {
    const mod = b.createModule(.{
        .root_source_file = b.path(root_source),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        .sanitize_c = .off,
    });

    const options = b.addOptions();
    options.addOption(bool, "probe_ported", false);
    options.addOption(bool, "probe_cpu", false);
    mod.addOptions("config", options);

    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/src"));
    // The CPU sources reach their own headers unqualified.
    mod.addIncludePath(b.path(root ++ "/ggml/src/ggml-cpu"));

    const defines = [_][]const u8{
        "-DGGML_USE_CPU",
        "-DGGML_SCHED_MAX_COPIES=4",
        "-DGGML_USE_ACCELERATE",
        "-DACCELERATE_NEW_LAPACK",
        "-DACCELERATE_LAPACK_ILP64",
        "-DGGML_USE_LLAMAFILE",
        "-DGGML_VERSION=\"0.3.0\"",
        "-DGGML_COMMIT=\"c1d0e7a00\"",
    };

    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = c_sources,
        .flags = &(c_flags ++ defines),
    });
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ported_test_cxx_sources,
        .flags = &(cxx_flags ++ defines),
    });

    if (opts.sdk) |sdk| mod.addFrameworkPath(.{ .cwd_relative = sdk });
    mod.linkFramework("Accelerate", .{});
    return mod;
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
