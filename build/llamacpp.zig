//! Build graph for the vendored llama.cpp reference tree.
//!
//! Replaces the CMake files under `llama.cpp/` for the macOS arm64
//! configuration this project targets, and for a CPU-only build off Apple
//! platforms (see `isApple`). The sources are compiled exactly as
//! upstream ships them — nothing under `llama.cpp/` is modified — so this file
//! is a translation of the build description, not of the code.
//!
//! The `-D` options that `make buildmacos` passes to CMake are hardcoded here:
//! static libraries, Metal on with the shader library embedded, Accelerate on,
//! BLAS and OpenMP off, no curl.

const std = @import("std");
const upstream = @import("upstream.zig");

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

/// Whether the target gets the Apple configuration: Metal, its Obj-C host
/// layer and embedded kernels, and Accelerate. Every Darwin target -- macOS,
/// iOS, the simulator -- does; anything else builds the CPU backend alone,
/// as upstream's CMake does when neither `GGML_METAL` nor `GGML_ACCELERATE`
/// finds its framework.
///
/// Parameters:
/// - `opts`: the build options; only the target is read.
///
/// Return: true for a Darwin target.
fn isApple(opts: Options) bool {
    return opts.target.result.os.tag.isDarwin();
}

/// Whether `src/ggml/varargs.c` must define ggml's variadic entry points,
/// because Zig cannot for this target. Must agree with `impl.zig_varargs`.
fn needsVarargsShim(opts: Options) bool {
    const t = opts.target.result;
    return switch (t.cpu.arch) {
        .aarch64, .aarch64_be => !(t.os.tag.isDarwin() or t.os.tag == .windows),
        .x86_64 => t.os.tag == .windows,
        else => false,
    };
}

/// Adds `src/ggml/varargs.c` to a module rooted in `src/ggml/` when the
/// target needs it; see `needsVarargsShim`.
fn addVarargsShim(b: *std.Build, mod: *std.Build.Module, opts: Options) void {
    if (!needsVarargsShim(opts)) return;
    mod.addCSourceFile(.{ .file = b.path("src/ggml/varargs.c"), .flags = &c_flags });
}

/// The `-D` flags `GGML_ACCELERATE` adds (ggml/src/ggml-cpu/CMakeLists.txt).
const accelerate_defines = [_][]const u8{
    "-DGGML_USE_ACCELERATE",
    "-DACCELERATE_NEW_LAPACK",
    "-DACCELERATE_LAPACK_ILP64",
};

/// Joins the flag lists into one slice owned by the build graph.
fn flags(b: *std.Build, parts: []const []const []const u8) []const []const u8 {
    return std.mem.concat(b.allocator, []const u8, parts) catch @panic("OOM");
}

/// Adds an Apple SDK's framework and library directories as search paths.
///
/// The SDK root itself is not a framework directory: Zig finds the macOS
/// frameworks on a native build without help, which hid that until an iOS
/// build searched `iPhoneOS.sdk/Foundation.framework` and failed. Taken from
/// `baseModule` in `llama.cpp.zmake/build.zig`.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the module to add the paths to.
/// - `sdk`: the SDK root; null adds nothing.
pub fn addSdkPaths(b: *std.Build, mod: *std.Build.Module, sdk: ?[]const u8) void {
    const path = sdk orelse return;
    mod.addFrameworkPath(b.graph.cwdRelativePath(b.pathJoin(&.{ path, "System/Library/Frameworks" })));
    mod.addLibraryPath(b.graph.cwdRelativePath(b.pathJoin(&.{ path, "usr/lib" })));
}

/// Writes a `zig libc` file describing an Apple SDK.
///
/// A named `-Dtarget` is a cross-compile to Zig even when it names the host,
/// and Zig then sets up no SDK include chain: its clang finds no `stdio.h`, and
/// its own libc++ build -- which sees none of our flags -- dies on `mbstate_t`.
/// A libc file is the one knob that reaches that internal build. Taken from
/// `writeAppleLibcFile` in `llama.cpp.zmake/build.zig`, where the same
/// problem was solved for iOS first.
///
/// Zig 0.17 requires `crt_dir` and `darwin_sdk_dir`; they get the values
/// `zig libc -target aarch64-ios` writes. The keys left empty belong to other
/// platforms.
///
/// Parameters:
/// - `b`: the build graph.
/// - `sdk`: the SDK root.
///
/// Return: a lazy path to the generated file, valid for the whole build.
pub fn writeAppleLibcFile(b: *std.Build, sdk: []const u8) std.Build.LazyPath {
    return b.addWriteFiles().add("apple-libc.txt", b.fmt(
        \\include_dir={s}/usr/include
        \\sys_include_dir={s}/usr/include
        \\cc_dir=
        \\crt_dir={s}/usr/lib
        \\msvc_lib_dir=
        \\kernel32_lib_dir=
        \\darwin_sdk_dir={s}
        \\
    , .{ sdk, sdk, sdk, sdk }));
}

/// Translates `src/ggml/c_import.h`, the ggml headers `src/ggml/impl.zig` imports.
///
/// Zig 0.17 has no `@cImport`, so the header list that file imported inline is
/// translated here and added, as `ggml_c`, to every module rooted in `src/ggml/`.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target and optimize mode.
///
/// Return: the translated module, owned by the build graph.
fn ggmlCModule(b: *std.Build, opts: Options) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/ggml/c_import.h"),
        .target = opts.target,
        .optimize = opts.optimize,
    });
    translate.addIncludePath(b.path(root ++ "/ggml/include"));
    translate.addIncludePath(b.path(root ++ "/ggml/src"));
    translate.addIncludePath(b.path(root ++ "/ggml/src/ggml-cpu"));
    return translate.createModule();
}

/// Translates `src/llama/c_import.h`, the headers `src/llama/impl.zig` imports.
///
/// Zig 0.17 has no `@cImport`; the result is added to libllama's module as
/// `llama_c`.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target and optimize mode.
///
/// Return: the translated module, owned by the build graph.
fn llamaCModule(b: *std.Build, opts: Options) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/llama/c_import.h"),
        .target = opts.target,
        .optimize = opts.optimize,
    });
    translate.addIncludePath(b.path(root ++ "/include"));
    translate.addIncludePath(b.path(root ++ "/ggml/include"));
    return translate.createModule();
}

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
    // ggml.cpp is dropped, not ported (PLAN.md, "Order of work"): it only
    // installs a std::terminate handler that prints a ggml backtrace when a
    // C++ exception escapes, and no C++ is left in the build to throw one.
    //
    // ggml-backend.cpp is ported: src/ggml/backend.zig and
    // src/ggml/backend_sched.zig. 102 C-ABI symbols, not the 82 PLAN.md
    // recorded; none of its 141 mangled exports is reached from another
    // object, so the Stage 3 swap applies unchanged.
    // ggml-backend-meta.cpp is ported: src/ggml/backend_meta.zig, gated by
    // scripts/meta-diff. The multi-device backend behind
    // LLAMA_SPLIT_MODE_TENSOR.
    //
    // ggml-opt.cpp is ported: src/ggml/opt.zig, gated by scripts/opt-diff.
    //
    // ggml-threading.cpp is ported: src/ggml/threading.zig. The first C++
    // translation unit to be swapped out.
    //
    // gguf.cpp is ported: src/ggml/gguf.zig. 61 C-ABI symbols -- the two the
    // C++ additionally exports, `gguf_type_size` and `gguf_write_to_buf`, are
    // declared in ggml-impl.h inside `#ifdef __cplusplus` for test code this
    // project does not build, so their symbols are mangled and outside the
    // contract.
    //
    // ggml-backend-reg.cpp is ported: src/ggml/backend_reg.zig. It took
    // ggml-backend-dl.cpp with it -- the registry was its only caller, and its
    // three `dl_*` wrappers have C++ linkage that Zig cannot provide.
};

const ggml_cpu_c_sources = [_][]const u8{
    // Empty: ggml-cpu.c, ggml-cpu/quants.c and ggml-cpu/arch/arm/quants.c are
    // all ported. **No C compiles anywhere under ggml/src/ any more.**
};

const ggml_cpu_cxx_sources = [_][]const u8{
    // ggml-cpu.cpp, traits.cpp, repack.cpp and arch/arm/repack.cpp used to
    // be here. They are ported now -- src/ggml/cpu/{cpu_backend,extra}.zig
    // and src/ggml/cpu/repack/ -- and had to move together, because
    // repack.cpp derives from the two abstract bases traits.cpp declares.
    // What is left are three translation units that are empty on this
    // target: hbm.cpp needs GGML_USE_CPU_HBM, the two amx files __AMX_INT8__.
    "ggml/src/ggml-cpu/hbm.cpp",
    "ggml/src/ggml-cpu/amx/amx.cpp",
    "ggml/src/ggml-cpu/amx/mmq.cpp",
};

const ggml_metal_cxx_sources = [_][]const u8{
    // ggml-metal.cpp is ported -- src/ggml/metal/backend.zig.
    // ggml-metal-device.cpp is ported -- src/ggml/metal/library.zig.
    // ggml-metal-common.cpp is ported -- src/ggml/metal/common.zig.
    // ggml-metal-ops.cpp is ported -- src/ggml/metal/ops.zig.
    // ggml-metal-tuning.cpp is ported -- src/ggml/metal/tuning.zig. Its
    // seven entry points are C++-linkage, so the Zig exports them under
    // their mangled names and the three C++ callers link unchanged.
    // ggml-metal-fusion.cpp is ported -- src/ggml/metal/fusion.zig.
};

const ggml_metal_objc_sources = [_][]const u8{
    "ggml/src/ggml-metal/ggml-metal-device.m",
    "ggml/src/ggml-metal/ggml-metal-context.m",
};

// Every translation unit under `src/` is ported to Zig: libllama swapped
// once, at `llama.h` (PLAN.md, Decision 42). `libllama.a` is now the Zig
// alone -- `src/llama/module.zig`, with `exports.zig` exporting the C API.
//
// The stock C++ is still compiled, as `libllama-ref.a`, for the in-process
// harnesses (`scripts/*-diff`), which link it beside the port and compare.
// It is the whole of upstream's libllama, the sampler and the grammar
// included, unmodified.
const llama_ref_sources = [_][]const u8{
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
    "src/llama-memory-hybrid-idx.cpp",
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
    /// libllama, linked against `ggml`: the port, all Zig.
    llama: *std.Build.Step.Compile,
    /// Upstream's libllama, the stock C++ unmodified, for the in-process
    /// harnesses that compare the port against it.
    llama_ref: *std.Build.Step.Compile,
    /// The ggml module, exposed so the ported Zig under `src/ggml/` can be
    /// unit-tested. Tests need the same C sources and frameworks the library
    /// does, because ported code calls back into the parts still in C.
    ggml_module: *std.Build.Module,
};

/// Options that shape the reference build.
pub const Options = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.lang.Optimize,
    /// Apple SDK root for the target -- macOS, iPhoneOS or iPhoneSimulator.
    /// Its frameworks and libraries are added as search paths; see
    /// `addSdkPaths`. Null off Apple platforms.
    sdk: ?[]const u8 = null,
    /// A `zig libc` file describing `sdk`, for a named `-Dtarget`. Null for a
    /// native build, where Zig sets up the SDK include chain itself. See
    /// `writeAppleLibcFile`.
    libc_file: ?std.Build.LazyPath = null,
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
    /// Builds upstream's CUDA backend into `libggml.a`; null leaves it out.
    /// See `addCuda`.
    cuda: ?Cuda = null,
    /// Builds ggml's Vulkan backend into `libggml.a`; null leaves it out.
    /// See `addVulkan`.
    vulkan: ?Vulkan = null,
    /// Builds ggml's BLAS backend, over Accelerate, into `libggml.a`; null
    /// leaves it out. Apple only (PLAN.md, Decision 60). See `addBlas`.
    blas: ?Blas = null,
};

/// ggml's BLAS backend, which this build compiles over Apple's Accelerate
/// alone (PLAN.md, Decision 60).
pub const Blas = struct {
    /// Use the Zig port (`src/ggml/blas.zig`) in place of upstream's
    /// `ggml-blas.cpp`. `build.zig` defaults it on; the gates build both.
    port: bool = true,
};

/// The Vulkan toolchain every Vulkan build uses, resolved by `build.zig`
/// (PLAN.md, Decisions 55 and 59).
pub const Vulkan = struct {
    /// The `glslc` every shader is compiled with.
    glslc: []const u8,
    /// Directories holding `vulkan/` and `spirv/`, searched after libc's.
    include_dirs: []const []const u8,
    /// Extra directories to find `libvulkan` in.
    lib_dirs: []const []const u8,
    /// `-D` flags for the shader extensions `glslc` accepts, from the
    /// probes `ggml-vulkan/CMakeLists.txt` runs.
    features: []const []const u8,
    /// Use the Zig host layer (`src/ggml/vulkan/`) in place of upstream's
    /// `ggml-vulkan.cpp`. Off until the swap; the gates build both.
    port: bool = false,
};

/// Where the CUDA toolkit is and which GPU to compile for.
pub const Cuda = struct {
    /// Toolkit root, e.g. `/usr/local/cuda`; `bin/nvcc` must exist under it.
    toolkit: []const u8,
    /// The `compute_`/`sm_` suffix to generate code for, e.g. `121a` for the
    /// GB10. Upstream's CMake rewrites every plain `12X` to `12Xa`, because
    /// the Blackwell FP4 tensor-core instructions need the
    /// architecture-specific variant; the build option does the same.
    arch: []const u8,
    /// The host compiler's `libstdc++.so`, which `nvcc`'s objects need. Linked
    /// by path: Zig reads a system library named `stdc++` as its own libc++.
    libstdcxx: []const u8,
    /// Use the Zig host layer (`src/ggml/cuda/`) in place of upstream's.
    /// Until the swap (Decision 48) this is off by default and the gates
    /// build both.
    port: bool = false,
    /// Upstream's `GGML_CUDA_FA_QUANTS`: the FlashAttention K-V type
    /// combinations whose vector kernels are compiled, `"all"` or a list
    /// such as `"q8_0-q8_0;q8_0-q4_0"`. See `cudaFaCombinations`.
    fa_quants: []const u8 = "all",
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
    const llama = addLlama(b, opts, ggml);
    const llama_ref = try addLlamaRef(b, opts, ggml);
    return .{ .ggml = ggml, .llama = llama, .llama_ref = llama_ref, .ggml_module = ggml.root_module };
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
    // Mirrors the `-DGGML_USE_METAL` below, for ported Zig that has to make
    // the same choice the C++ preprocessor does. `src/ggml/backend_reg.zig`
    // is the one that needs it: the registry constructor registers Metal
    // behind `#ifdef GGML_USE_METAL`, and the ported test root links no
    // Metal at all.
    options.addOption(bool, "use_metal", isApple(opts));
    // Mirrors the `-DGGML_USE_CUDA` `addCuda` defines, for the same reason.
    options.addOption(bool, "use_cuda", opts.cuda != null);
    options.addOption(bool, "cuda_port", if (opts.cuda) |cu| cu.port else false);
    // The fatbins' architecture, for `ggml_cuda_highest_compiled_arch`.
    options.addOption([]const u8, "cuda_arch", if (opts.cuda) |cu| cu.arch else "");
    // `GGML_CUDA_FA_QUANTS` as the C sees it, and the K-V combinations it
    // compiles, for `src/ggml/cuda/fattn.zig` and `backend.zig`.
    const fa: CudaFa = if (opts.cuda) |cu| cudaFaCombinations(b, cu) else .{ .define = "", .combinations = &.{} };
    options.addOption([]const u8, "cuda_fa_quants", fa.define);
    options.addOption([]const []const u8, "cuda_fa_combinations", fa.combinations);
    // Mirrors `-DGGML_USE_VULKAN`, for `backend_reg.zig`.
    options.addOption(bool, "use_vulkan", opts.vulkan != null);
    options.addOption(bool, "vulkan_port", if (opts.vulkan) |vk| vk.port else false);
    // Mirrors `-DGGML_USE_BLAS`, for `backend_reg.zig`.
    options.addOption(bool, "use_blas", opts.blas != null);
    options.addOption(bool, "blas_port", if (opts.blas) |bl| bl.port else false);
    // The shader extensions `glslc` accepted: the C++'s
    // `GGML_VULKAN_*_GLSLC_SUPPORT` macros, for the ported host layer.
    for (vulkan_glslc_features) |f| {
        options.addOption(bool, f.option, if (opts.vulkan) |vk| hasDefine(vk.features, f.define) else false);
    }
    mod.addOptions("config", options);
    mod.addImport("ggml_c", ggmlCModule(b, opts));
    addVarargsShim(b, mod, opts);

    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/src"));
    mod.addIncludePath(b.path(root ++ "/ggml/src/ggml-cpu"));
    mod.addIncludePath(b.path(root ++ "/ggml/src/ggml-metal"));

    // Backend selection. Upstream sets these from the -D options; this build
    // targets one configuration per platform, so they are fixed.
    const apple_defines = [_][]const u8{"-DGGML_USE_METAL"} ++ accelerate_defines ++
        [_][]const u8{"-DGGML_METAL_EMBED_LIBRARY"};
    const common_defines = [_][]const u8{
        "-DGGML_USE_CPU",
        "-DGGML_SCHED_MAX_COPIES=4",
        "-DGGML_USE_LLAMAFILE",
        // Upstream's CMake defaults GGML_CPU_REPACK to ON
        // (ggml/CMakeLists.txt:152). Without it `ggml-cpu.cpp` never
        // registers the repack buffer type, the extra-buffer list is
        // empty, and `repack.cpp` plus `arch/arm/repack.cpp` compile
        // into the library unreachable -- 5,692 live lines of dead
        // code. Measured, by panicking on the one entry point.
        "-DGGML_USE_CPU_REPACK",
    };
    const defines = flags(b, &.{ &common_defines, if (isApple(opts)) &apple_defines else &.{} });

    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ggml_base_sources ++ &ggml_cpu_c_sources,
        .flags = flags(b, &.{ &c_flags, defines }),
    });

    // ggml.c is ported and swapped out: see src/ggml/{impl,types,context,
    // runtime,ops,graph,quantize}.zig, reached through the module barrel this
    // library is rooted at. Nothing compiles it any more, so the linker's
    // missing-symbol errors are what prove the port is complete.
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ggml_base_cxx_sources ++ &ggml_cpu_cxx_sources ++ &ggml_metal_cxx_sources,
        .flags = flags(b, &.{ &cxx_flags, defines }),
    });

    if (isApple(opts)) {
        mod.addCSourceFiles(.{
            .root = b.path(root),
            .files = &ggml_metal_objc_sources,
            .flags = flags(b, &.{ &objc_flags, defines }),
        });

        for (metal_kernels) |kind| {
            mod.addAssemblyFile(embedMetalKernel(b, kind));
        }

        addSdkPaths(b, mod, opts.sdk);
        mod.linkFramework("Foundation", .{});
        mod.linkFramework("Metal", .{});
        mod.linkFramework("MetalKit", .{});
        mod.linkFramework("Accelerate", .{});
    }

    if (opts.cuda) |cuda| try addCuda(b, mod, cuda);
    if (opts.vulkan) |vk| try addVulkan(b, mod, vk);
    if (opts.blas) |bl| addBlas(b, mod, bl, defines);

    // LLVM's backend, always. The port calls LLVM intrinsics by name --
    // `sdot` and `smmla` on ARM, `vcvtps2ph` on x86 -- which only that
    // backend understands. Zig 0.17's own x86_64 backend, its Debug default,
    // emits them as undefined external symbols instead. Measured: the x86
    // Debug link failed on `llvm.x86.vcvtps2ph.256`.
    const lib = b.addLibrary(.{ .name = "ggml", .root_module = mod, .linkage = .static, .use_llvm = true });
    lib.setLibCFile(opts.libc_file);
    return lib;
}

/// The `-D` flags upstream's CMake gives every `ggml-cuda` translation unit
/// for the configuration this build selects: `GGML_CUDA_GRAPHS` on, NCCL off
/// (one GPU). Taken from the `flags.make` a stock CMake build of the pinned
/// tree writes, minus the `GGML_BACKEND_SHARED`/`GGML_SHARED` pair, which
/// only exist to export symbols from a `.so`, and minus the FlashAttention
/// type defines, which depend on `Cuda.fa_quants` (`cudaFaDefines`).
const cuda_defines = [_][]const u8{
    "-DGGML_CUDA_USE_GRAPHS",
    "-DGGML_SCHED_MAX_COPIES=4",
    "-D_GNU_SOURCE",
    "-D_XOPEN_SOURCE=600",
    "-DNDEBUG",
};

/// The K and V types a FlashAttention vector kernel can be instantiated
/// for, in the order `ggml_cuda_fattn_vec_instances`
/// (ggml/cmake/common.cmake:53 @b29c606e2) iterates them: its `FA_TYPES`.
const cuda_fa_types = [_][]const u8{ "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "bf16", "f16" };

/// What `cudaFaCombinations` derives from `Cuda.fa_quants`.
const CudaFa = struct {
    /// The value `GGML_CUDA_FA_QUANTS` is defined to.
    define: []const u8,
    /// Every compiled K-V combination, spelled `"<k>-<v>"` as the instance
    /// files are.
    combinations: []const []const u8,
};

/// Mirrors `ggml_cuda_fattn_vec_instances` (ggml/cmake/common.cmake:53
/// @b29c606e2): the K-V type combinations whose FlashAttention vector
/// kernel instances `GGML_CUDA_FA_QUANTS` selects. `"all"` is every pair,
/// V outermost; otherwise `f16-f16` and each listed combination, separated
/// by `;` or `,`, duplicates dropped. A malformed list fails the build, as
/// it fails CMake's configure.
///
/// Parameters:
/// - `b`: the build graph; the result is allocated on it.
/// - `cuda`: carries `fa_quants`.
///
/// Return: the define's value and the combinations.
fn cudaFaCombinations(b: *std.Build, cuda: Cuda) CudaFa {
    const lowered = b.allocator.alloc(u8, cuda.fa_quants.len) catch @panic("OOM");
    _ = std.ascii.lowerString(lowered, cuda.fa_quants);
    const fa_quants = std.mem.trim(u8, lowered, " \t\r\n");
    if (fa_quants.len == 0) std.debug.panic("GGML_CUDA_FA_QUANTS must not be empty", .{});

    var combinations: std.ArrayList([]const u8) = .empty;
    if (std.mem.eql(u8, fa_quants, "all")) {
        for (cuda_fa_types) |type_V| {
            for (cuda_fa_types) |type_K| {
                combinations.append(b.allocator, b.fmt("{s}-{s}", .{ type_K, type_V })) catch @panic("OOM");
            }
        }
    } else {
        combinations.append(b.allocator, "f16-f16") catch @panic("OOM");
        var it = std.mem.tokenizeAny(u8, fa_quants, ";,");
        while (it.next()) |raw| {
            const combination = std.mem.trim(u8, raw, " \t\r\n");
            const dash = std.mem.findScalar(u8, combination, '-') orelse
                std.debug.panic("GGML_CUDA_FA_QUANTS: \"{s}\" is not \"all\" or a <type_K>-<type_V> combination", .{combination});
            for ([_][]const u8{ combination[0..dash], combination[dash + 1 ..] }) |ty| {
                const known = for (cuda_fa_types) |t| {
                    if (std.mem.eql(u8, t, ty)) break true;
                } else false;
                if (!known) std.debug.panic("GGML_CUDA_FA_QUANTS: unknown type \"{s}\" in \"{s}\"", .{ ty, combination });
            }
            const seen = for (combinations.items) |have| {
                if (std.mem.eql(u8, have, combination)) break true;
            } else false;
            if (!seen) combinations.append(b.allocator, combination) catch @panic("OOM");
        }
    }

    const define = b.allocator.dupe(u8, fa_quants) catch @panic("OOM");
    std.mem.replaceScalar(u8, define, ';', ',');
    return .{ .define = define, .combinations = combinations.items };
}

/// The FlashAttention `-D` flags `ggml_cuda_fattn_vec_instances` adds to
/// every `ggml-cuda` translation unit: `GGML_CUDA_FA_QUANTS` as a string,
/// and `GGML_CUDA_FA_<K>_<V>` as 1 or 0 for every type pair, which
/// `ggml_cuda_get_fattn_vec_case`'s `if constexpr` reads.
///
/// Parameters:
/// - `b`: the build graph; the result is allocated on it.
/// - `cuda`: carries `fa_quants`.
///
/// Return: the flags, in CMake's order.
fn cudaFaDefines(b: *std.Build, cuda: Cuda) []const []const u8 {
    const fa = cudaFaCombinations(b, cuda);
    var defines: std.ArrayList([]const u8) = .empty;
    defines.append(b.allocator, b.fmt("-DGGML_CUDA_FA_QUANTS=\"{s}\"", .{fa.define})) catch @panic("OOM");
    for (cuda_fa_types) |type_V| {
        for (cuda_fa_types) |type_K| {
            const name = b.fmt("{s}-{s}", .{ type_K, type_V });
            const compiled = for (fa.combinations) |have| {
                if (std.mem.eql(u8, have, name)) break true;
            } else false;
            const def = b.fmt("-DGGML_CUDA_FA_{s}_{s}={d}", .{ type_K, type_V, @intFromBool(compiled) });
            _ = std.ascii.upperString(def["-D".len .. def.len - 2], def["-D".len .. def.len - 2]);
            defines.append(b.allocator, def) catch @panic("OOM");
        }
    }
    return defines.items;
}

/// Compiles upstream's `ggml-cuda` with `nvcc` and adds it to the ggml module.
///
/// **This is the reference CUDA backend, not a port.** PLAN.md's Stage 6 has
/// the Zig host layer calling `cudaLaunchKernel`; until that exists, the
/// whole of `ggml/src/ggml-cuda/` -- host code included -- compiles unmodified,
/// the position `llama.cpp.zmake` holds for the rest of the tree. Its
/// external contract is C: every ggml symbol its objects leave undefined is a
/// C-ABI export of the Zig ggml (measured with `nm -u` over all 187 objects),
/// and it registers through `ggml_backend_cuda_reg`, which
/// `src/ggml/backend_reg.zig` calls under `config.use_cuda`.
///
/// The source set is upstream's `CMakeLists.txt`'s: every `*.cu`, plus the
/// `fattn-tile`, `fattn-mma`, `mmq` and `mmf` template instances, and the
/// `fattn-vec` ones `GGML_CUDA_FA_QUANTS` selects. One `nvcc` run
/// per file, so the build graph parallelises and caches them individually; a
/// clean build is ~2 minutes on the GB10's 20 cores.
///
/// `nvcc`'s host compiler is the system `g++`, so the objects need
/// libstdc++, which is linked beside the CUDA runtime, cuBLAS and the driver
/// (for VMM). They are compiled `-mno-outline-atomics`: g++ on aarch64
/// otherwise calls libgcc's `__aarch64_cas4_sync` and friends, which Zig's
/// compiler_rt does not define and Zig's linker does not pull from libgcc.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the ggml module the objects and libraries are added to.
/// - `cuda`: toolkit root and target architecture.
///
/// Return: nothing; propagates errors from reading the source directory.
fn addCuda(b: *std.Build, mod: *std.Build.Module, cuda: Cuda) !void {
    const files = try cudaSources(b, cuda);
    if (cuda.port) {
        addCudaPort(b, mod, cuda, files);
    } else {
        for (files) |file| mod.addObjectFile(nvccCompile(b, cuda, file, .object, &.{}));
    }
    addCudaLibraries(b, mod, cuda);
}

/// Upstream's `ggml-cuda` source set, sorted.
///
/// The set is upstream's `CMakeLists.txt`'s: every `*.cu`, plus the
/// `fattn-tile`, `fattn-mma`, `mmq` and `mmf` template instances, and the
/// `fattn-vec` instances of the combinations `cudaFaCombinations` selects.
///
/// Parameters:
/// - `b`: the build graph; the returned paths are allocated on it.
/// - `cuda`: carries `fa_quants`.
///
/// Return: paths relative to the build root; propagates directory errors.
fn cudaSources(b: *std.Build, cuda: Cuda) ![]const []const u8 {
    const dir_path = root ++ "/ggml/src/ggml-cuda";
    var files: std.ArrayList([]const u8) = .empty;
    const instance_prefixes = [_][]const u8{ "fattn-tile", "fattn-mma", "mmq", "mmf" };
    const fa_combinations = cudaFaCombinations(b, cuda).combinations;
    var fa_found: usize = 0;
    for ([_][]const u8{ "", "/template-instances" }) |sub| {
        const path = b.fmt("{s}{s}", .{ dir_path, sub });
        b.dependOnDirectoryMetadata(b.path(path));
        var dir = try b.root.openDir(b.graph.io, path, .{ .iterate = true });
        defer dir.close(b.graph.io);
        var it = dir.iterate();
        while (try it.next(b.graph.io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".cu")) continue;
            if (sub.len != 0) {
                const vec_prefix = "fattn-vec-instance-";
                const wanted = if (std.mem.startsWith(u8, entry.name, vec_prefix)) vec: {
                    const combination = entry.name[vec_prefix.len .. entry.name.len - ".cu".len];
                    for (fa_combinations) |have| {
                        if (std.mem.eql(u8, have, combination)) {
                            fa_found += 1;
                            break :vec true;
                        }
                    }
                    break :vec false;
                } else for (instance_prefixes) |prefix| {
                    if (std.mem.startsWith(u8, entry.name, prefix)) break true;
                } else false;
                if (!wanted) continue;
            }
            try files.append(b.allocator, b.fmt("{s}/{s}", .{ path, entry.name }));
        }
    }
    // CMake fails when a combination has no instance file.
    if (fa_found != fa_combinations.len) {
        std.debug.panic("GGML_CUDA_FA_QUANTS: {d} of {d} FlashAttention template instances exist", .{ fa_found, fa_combinations.len });
    }
    // Directory order is the filesystem's; sort so the archive is stable.
    std.mem.sort([]const u8, files.items, {}, struct {
        fn lessThan(_: void, l: []const u8, r: []const u8) bool {
            return std.mem.lessThan(u8, l, r);
        }
    }.lessThan);
    return files.items;
}

/// What one `nvcc` run produces.
const NvccOutput = enum {
    /// `-c`: a host object with the device code registered through cudart.
    /// The reference.
    object,
    /// `-fatbin`: device code alone, for the Zig host layer to load
    /// (Decision 44).
    fatbin,
};

/// Runs `nvcc` on one upstream translation unit with the reference's flags.
///
/// Parameters:
/// - `b`: the build graph.
/// - `cuda`: toolkit root and target architecture.
/// - `file`: the `.cu` source, relative to the build root.
/// - `kind`: object or fatbin.
/// - `extra`: further arguments, e.g. the `-D` renames of `addCudaRef`.
///
/// Return: the output file, valid for the build's lifetime.
fn nvccCompile(b: *std.Build, cuda: Cuda, file: []const u8, kind: NvccOutput, extra: []const []const u8) std.Build.LazyPath {
    const dir_path = root ++ "/ggml/src/ggml-cuda";
    const nvcc = b.pathJoin(&.{ cuda.toolkit, "bin", "nvcc" });
    const gencode = b.fmt("--generate-code=arch=compute_{s},code=[compute_{s},sm_{s}]", .{ cuda.arch, cuda.arch, cuda.arch });
    const mode = switch (kind) {
        .object => "-c",
        .fatbin => "-fatbin",
    };
    const run = b.addSystemCommand(&.{ nvcc, mode, "-O3", "-std=c++17", gencode, "-use_fast_math", "-extended-lambda", "-compress-mode=size", "-Xcompiler=-Wno-pedantic,-mno-outline-atomics" });
    run.addArgs(&cuda_defines);
    run.addArgs(cudaFaDefines(b, cuda));
    run.addArgs(extra);
    run.addDirectoryArg2(b.path(root ++ "/ggml/include"), .{ .prefix = "-I" });
    run.addDirectoryArg2(b.path(root ++ "/ggml/src"), .{ .prefix = "-I" });
    run.addDirectoryArg2(b.path(dir_path), .{ .prefix = "-I" });
    // nvcc writes a make-style depfile, so editing a `.cuh` rebuilds
    // exactly the outputs that include it.
    // `-MF` must be its own argument: nvcc rejects the joined form.
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2("out.d", .{});
    run.addFileArg(b.path(file));
    run.addArg("-o");
    const ext = switch (kind) {
        .object => "o",
        .fatbin => "fatbin",
    };
    return run.addOutputFileArg2(b.fmt("{s}.{s}", .{ std.fs.path.stem(file), ext }), .{});
}

/// Adds the Zig CUDA host layer's inputs to the ggml module: the fatbins,
/// the kernel table generated from them, and the translated CUDA headers.
///
/// Every upstream translation unit becomes a fatbin (Decision 44). Its entry
/// symbols are listed with `cuobjdump` and demangled with `cu++filt`, and
/// `build/cuda_kernels.zig` turns the listings into the `cuda_kernels`
/// module, which embeds the fatbins (Decision 45). No upstream host code is
/// linked.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the ggml module.
/// - `cuda`: toolkit root and target architecture.
/// - `files`: `cudaSources`'s list.
fn addCudaPort(b: *std.Build, mod: *std.Build.Module, cuda: Cuda, files: []const []const u8) void {
    const bin = b.pathJoin(&.{ cuda.toolkit, "bin" });
    const tool = b.addExecutable(.{
        .name = "cuda_kernels",
        .root_module = b.createModule(.{
            .root_source_file = b.path("build/cuda_kernels.zig"),
            .target = b.graph.host,
            .optimize = .debug,
        }),
    });
    const gen = b.addRunArtifact(tool);
    const table = gen.addOutputFileArg("cuda_kernels.zig");

    var fatbins: std.ArrayList(std.Build.LazyPath) = .empty;
    for (files) |file| {
        const fatbin = nvccCompile(b, cuda, file, .fatbin, &.{});
        fatbins.append(b.allocator, fatbin) catch @panic("OOM");

        // One entry symbol per line, then the same lines demangled beside
        // them. Only entry points (`STO_ENTRY`) are kernels; the rest are
        // device functions and data.
        const list = b.addSystemCommand(&.{
            "/bin/sh", "-c",
            \\set -e
            \\"$1/cuobjdump" -symbols "$2" | awk '/STO_ENTRY/ { print $NF }' | sort -u > "$3.mangled"
            \\"$1/cu++filt" < "$3.mangled" | paste "$3.mangled" - > "$3"
            ,
            "sh",      bin,
        });
        list.addFileArg(fatbin);
        const listing = list.addOutputFileArg(b.fmt("{s}.symbols", .{std.fs.path.stem(file)}));

        gen.addArg(std.fs.path.stem(file));
        gen.addFileArg(listing);
    }

    const kernels = b.createModule(.{ .root_source_file = table });
    for (fatbins.items, 0..) |fatbin, i| {
        kernels.addAnonymousImport(b.fmt("cuda_fatbin_{d}", .{i}), .{ .root_source_file = fatbin });
    }
    mod.addImport("cuda_kernels", kernels);
    mod.addImport("cuda_c", cudaCModule(b, mod, cuda));

    // Decision 47: CUB's device-wide sort and scan behind a C ABI. A host
    // object with its device code registered through cudart, like the
    // reference's, compiled with the reference's flags so its CUB
    // instantiations are the ones `argsort.cu` and `cumsum.cu` make.
    mod.addObjectFile(nvccCompile(b, cuda, "src/ggml/cuda/cub_shim.cu", .object, &.{}));
}

/// Translates `src/ggml/cuda/c_import.h`: the driver API, the runtime API
/// and cuBLAS.
///
/// A second translation beside `ggml_c` is safe here, where it would not be
/// for a ggml header: none of these headers includes ggml's, so no type
/// appears in both and none can come out twice incompatibly.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the module whose target and optimize mode are used.
/// - `cuda`: the toolkit root, whose `include/` is searched.
///
/// Return: the translated module, owned by the build graph.
fn cudaCModule(b: *std.Build, mod: *std.Build.Module, cuda: Cuda) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/ggml/cuda/c_import.h"),
        .target = mod.resolved_target.?,
        .optimize = mod.optimize.?,
    });
    translate.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ cuda.toolkit, "include" }) });
    return translate.createModule();
}

/// The `ggml-cuda.h` functions upstream's objects export unmangled --
/// measured with `nm -g --defined-only` over all 192 objects: these ten, all
/// in `ggml-cuda.o`, and nothing else.
pub const cuda_c_exports = [_][]const u8{
    "ggml_backend_cuda_buffer_type",
    "ggml_backend_cuda_get_device_count",
    "ggml_backend_cuda_get_device_description",
    "ggml_backend_cuda_get_device_memory",
    "ggml_backend_cuda_host_buffer_type",
    "ggml_backend_cuda_init",
    "ggml_backend_cuda_reg",
    "ggml_backend_cuda_register_host_buffer",
    "ggml_backend_cuda_unregister_host_buffer",
    "ggml_backend_is_cuda",
};

/// Builds upstream's CUDA backend a second time, with `cuda_c_exports`
/// renamed to `ref_*`, as `libggml-cuda-ref.a`.
///
/// For the in-process gates (`cuda-ops-diff`): the reference and the port
/// are registered side by side in one process and asked the same
/// questions. Renaming on the command line renames the definitions and
/// every internal caller alike, which is what `repack-diff` relies on too.
/// The objects' remaining exports are C++-mangled and the Zig port defines
/// none of them, so nothing else collides.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target and optimize mode; `opts.cuda` must be set.
///
/// Return: the library; propagates directory errors.
pub fn addCudaRef(b: *std.Build, opts: Options) !*std.Build.Step.Compile {
    const cuda = opts.cuda.?;
    const mod = b.createModule(.{ .target = opts.target, .optimize = opts.optimize, .link_libc = true });
    var renames: std.ArrayList([]const u8) = .empty;
    for (cuda_c_exports) |name| try renames.append(b.allocator, b.fmt("-D{s}=ref_{s}", .{ name, name }));
    for (try cudaSources(b, cuda)) |file| mod.addObjectFile(nvccCompile(b, cuda, file, .object, renames.items));
    return b.addLibrary(.{ .name = "ggml-cuda-ref", .root_module = mod, .linkage = .static });
}

/// Adds the CUDA runtime, cuBLAS, the driver and libstdc++ to a module's link.
///
/// The ggml module declares them, but a search path does not travel with a
/// static library to the executable that links it, so every module that
/// links `libggml.a` built with CUDA calls this too. The toolkit's `lib64` is
/// also added as an rpath, so the binary runs without `LD_LIBRARY_PATH` where
/// the loader's cache does not already list it.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the module whose link gains the libraries.
/// - `cuda`: the toolkit root; null adds nothing.
pub fn addCudaLibraries(b: *std.Build, mod: *std.Build.Module, cuda: ?Cuda) void {
    const tk = cuda orelse return;
    addCudaRuntime(b, mod, tk);
    mod.addObjectFile(.{ .cwd_relative = tk.libstdcxx });
}

/// `addCudaLibraries` without libstdc++, for an executable of C++ that
/// throws: upstream's tools and the libmtmd gates. Zig's libc++abi supplies
/// `__cxa_throw` while a libstdc++ on the line supplies
/// `__gxx_personality_v0`, and that personality aborts on Zig's exceptions,
/// so no exception can be caught -- upstream's argument parser reports every
/// bad flag by throwing. Zig's libc++ supplies all the CUB shim needs
/// (`scripts/platform-libs`), and libcublas loads libstdc++ itself at run
/// time.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the module whose link gains the libraries.
/// - `cuda`: the toolkit root; null adds nothing.
pub fn addCudaRuntime(b: *std.Build, mod: *std.Build.Module, cuda: ?Cuda) void {
    const tk = cuda orelse return;
    const lib64 = b.pathJoin(&.{ tk.toolkit, "lib64" });
    mod.addLibraryPath(.{ .cwd_relative = lib64 });
    mod.addRPath(.{ .cwd_relative = lib64 });
    for ([_][]const u8{ "cudart", "cublas", "cublasLt", "cuda" }) |name| {
        mod.linkSystemLibrary(name, .{});
    }
}

/// The `GGML_VULKAN_*_GLSLC_SUPPORT` macros and the `config` options that
/// carry them to the Zig host layer.
const vulkan_glslc_features = [_]struct { define: []const u8, option: []const u8 }{
    .{ .define = "-DGGML_VULKAN_COOPMAT_GLSLC_SUPPORT", .option = "vk_coopmat_glslc" },
    .{ .define = "-DGGML_VULKAN_COOPMAT2_GLSLC_SUPPORT", .option = "vk_coopmat2_glslc" },
    .{ .define = "-DGGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT", .option = "vk_coopmat2_decode_vector_glslc" },
    .{ .define = "-DGGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT", .option = "vk_integer_dot_glslc" },
    .{ .define = "-DGGML_VULKAN_BFLOAT16_GLSLC_SUPPORT", .option = "vk_bfloat16_glslc" },
    .{ .define = "-DGGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT", .option = "vk_float_e2m1_glslc" },
    .{ .define = "-DGGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT", .option = "vk_float_e4m3_glslc" },
};

fn hasDefine(defines: []const []const u8, define: []const u8) bool {
    for (defines) |d| if (std.mem.eql(u8, d, define)) return true;
    return false;
}

/// The flags `ggml-vulkan.cpp` compiles with in the reference: ggml's own,
/// `GGML_USE_VULKAN`, and the extension macros. `NDEBUG` is upstream's
/// release default, as for CUDA.
fn vulkanCxxFlags(b: *std.Build, vk: Vulkan, extra: []const []const u8) []const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    list.appendSlice(b.allocator, &cxx_flags) catch @panic("OOM");
    list.appendSlice(b.allocator, &.{ "-DGGML_USE_VULKAN", "-DGGML_SCHED_MAX_COPIES=4", "-DNDEBUG" }) catch @panic("OOM");
    list.appendSlice(b.allocator, vk.features) catch @panic("OOM");
    list.appendSlice(b.allocator, extra) catch @panic("OOM");
    // `-idirafter`, not `-I`: the header directory may be `/usr/include`,
    // which must not shadow the libc headers Zig supplies.
    for (vk.include_dirs) |dir| list.appendSlice(b.allocator, &.{ "-idirafter", dir }) catch @panic("OOM");
    return list.items;
}

/// What one generation of the shaders produced.
const VulkanShaders = struct {
    /// The directory holding `ggml-vulkan-shaders.hpp`.
    header_dir: std.Build.LazyPath,
    /// One generated `.cpp` per shader file, holding its variants' SPIR-V.
    cpps: []const std.Build.LazyPath,
};

/// The build's one shader generation, shared by every artifact that needs
/// it. See `vulkanShaders`.
var vulkan_shaders_once: ?VulkanShaders = null;

/// Runs upstream's `vulkan-shaders-gen` over every shader file, once per build
/// graph, and returns its output.
///
/// The steps are those of `llama.cpp.zmake`'s `addVulkan`, which replaces
/// CMake's `ExternalProject` (`ggml-vulkan/CMakeLists.txt:149-259`): the
/// generator built for the host with the extension macros, run once with no
/// source for the header and once per `.comp` with `--glslc`. Every `.comp`
/// and `.glsl` is an input of every run, because `glslc` resolves the
/// includes and Zig cannot see them.
///
/// **Once** is the point (PLAN.md, Decision 59): `glslc -O` is not
/// reproducible on `tri_f16`, so the reference objects and the port's
/// embedded table must come from the same run's files. Every caller in one
/// build graph gets the same `LazyPath`s.
///
/// Parameters:
/// - `b`: the build graph.
/// - `vk`: the toolchain.
///
/// Return: the generated header directory and `.cpp` files; propagates
/// directory-read failures.
fn vulkanShaders(b: *std.Build, vk: Vulkan) !VulkanShaders {
    if (vulkan_shaders_once) |done| return done;
    const shaders_dir = root ++ "/ggml/src/ggml-vulkan/vulkan-shaders";

    const gen_mod = b.createModule(.{
        .target = b.graph.host,
        .optimize = .fast,
        .link_libc = true,
        .link_libcpp = true,
        .sanitize_c = .off,
    });
    var gen_flags: std.ArrayList([]const u8) = .empty;
    try gen_flags.appendSlice(b.allocator, &cxx_flags);
    try gen_flags.appendSlice(b.allocator, vk.features);
    gen_mod.addCSourceFile(.{ .file = b.path(shaders_dir ++ "/vulkan-shaders-gen.cpp"), .flags = gen_flags.items });
    const gen = b.addExecutable(.{ .name = "vulkan-shaders-gen", .root_module = gen_mod });

    var comp: std.ArrayList([]const u8) = .empty;
    var glsl: std.ArrayList([]const u8) = .empty;
    {
        b.dependOnDirectoryMetadata(b.path(shaders_dir));
        var dir = try b.root.openDir(b.graph.io, shaders_dir, .{ .iterate = true });
        defer dir.close(b.graph.io);
        var it = dir.iterate();
        while (try it.next(b.graph.io)) |entry| {
            if (entry.kind != .file) continue;
            const path = b.fmt("{s}/{s}", .{ shaders_dir, entry.name });
            if (std.mem.endsWith(u8, entry.name, ".comp")) try comp.append(b.allocator, path);
            if (std.mem.endsWith(u8, entry.name, ".glsl")) try glsl.append(b.allocator, path);
        }
    }
    const lessThan = struct {
        fn f(_: void, l: []const u8, r: []const u8) bool {
            return std.mem.lessThan(u8, l, r);
        }
    }.f;
    std.mem.sort([]const u8, comp.items, {}, lessThan);
    std.mem.sort([]const u8, glsl.items, {}, lessThan);

    // Pass 1: the header, which names every variant.
    const hdr_run = b.addRunArtifact(gen);
    hdr_run.addArg("--output-dir");
    _ = hdr_run.addOutputDirectoryArg("spv");
    hdr_run.addArg("--target-hpp");
    const header = hdr_run.addOutputFileArg("ggml-vulkan-shaders.hpp");

    // Pass 2: one run per shader file, each writing that file's SPIR-V.
    var cpps: std.ArrayList(std.Build.LazyPath) = .empty;
    for (comp.items) |file| {
        const run = b.addRunArtifact(gen);
        run.addArgs(&.{ "--glslc", vk.glslc });
        run.addArg("--source");
        run.addFileArg(b.path(file));
        run.addArg("--output-dir");
        _ = run.addOutputDirectoryArg("spv");
        // Only its basename is used, for the generated `#include`.
        run.addArgs(&.{ "--target-hpp", "ggml-vulkan-shaders.hpp" });
        run.addArg("--target-cpp");
        try cpps.append(b.allocator, run.addOutputFileArg(b.fmt("{s}.cpp", .{std.fs.path.basename(file)})));
        for (comp.items) |f| run.addFileInput(b.path(f));
        for (glsl.items) |f| run.addFileInput(b.path(f));
    }

    vulkan_shaders_once = .{ .header_dir = header.dirname(), .cpps = cpps.items };
    return vulkan_shaders_once.?;
}

/// The `-D` flags `ggml-blas/CMakeLists.txt` adds for `GGML_BLAS_VENDOR`
/// Apple.
const blas_accelerate_defines = [_][]const u8{
    "-DACCELERATE_NEW_LAPACK",
    "-DACCELERATE_LAPACK_ILP64",
    "-DGGML_BLAS_USE_ACCELERATE",
};

/// Adds ggml's BLAS backend to the ggml module.
///
/// With `bl.port` off this is the reference: upstream's `ggml-blas.cpp`,
/// compiled unmodified with the Accelerate vendor's flags. With it on,
/// `src/ggml/blas.zig` takes its place through the module barrel, and nothing
/// is added here. Accelerate is already linked on every Darwin target.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the ggml module.
/// - `bl`: which side to build.
/// - `defines`: ggml's own `-D` flags for this target.
///
/// Return: nothing.
fn addBlas(b: *std.Build, mod: *std.Build.Module, bl: Blas, defines: []const []const u8) void {
    if (bl.port) return;
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &.{"ggml/src/ggml-blas/ggml-blas.cpp"},
        .flags = flags(b, &.{ &cxx_flags, defines, &blas_accelerate_defines }),
    });
}

/// Adds ggml's Vulkan backend to the ggml module.
///
/// With `vk.port` off this is the reference: upstream's `ggml-vulkan.cpp`
/// and the generated shader `.cpp` files, compiled unmodified. With it on,
/// the Zig host layer under `src/ggml/vulkan/` takes their place, and the
/// same generated files reach it as data through `addVulkanPort`.
///
/// `libvulkan` is not linked here: Zig resolves a system library even for a
/// static archive, and a machine that only builds the archive need not have
/// the loader. `addVulkanLibraries` links it where an executable is made.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the ggml module.
/// - `vk`: the toolchain.
///
/// Return: nothing; propagates directory-read failures.
fn addVulkan(b: *std.Build, mod: *std.Build.Module, vk: Vulkan) !void {
    const shaders = try vulkanShaders(b, vk);
    if (vk.port) {
        addVulkanPort(b, mod, vk, shaders);
        return;
    }
    mod.addIncludePath(shaders.header_dir);
    for (shaders.cpps) |cpp| mod.addCSourceFile(.{ .file = cpp, .flags = &cxx_flags });
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &.{"ggml/src/ggml-vulkan/ggml-vulkan.cpp"},
        .flags = vulkanCxxFlags(b, vk, &.{}),
    });
}

/// Adds the Zig Vulkan host layer's inputs to the ggml module: the shader
/// table `build/vulkan_shaders.zig` makes from the generated `.cpp` files,
/// and the translated Vulkan C headers.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the ggml module.
/// - `vk`: the toolchain.
/// - `shaders`: the build's one generation.
fn addVulkanPort(b: *std.Build, mod: *std.Build.Module, vk: Vulkan, shaders: VulkanShaders) void {
    const tool = b.addExecutable(.{
        .name = "vulkan_shaders",
        .root_module = b.createModule(.{
            .root_source_file = b.path("build/vulkan_shaders.zig"),
            .target = b.graph.host,
            // The largest generated file is 129 MB of hex.
            .optimize = .fast,
        }),
    });
    const run = b.addRunArtifact(tool);
    const table = run.addOutputFileArg("vulkan_shaders.zig");
    const blob = run.addOutputFileArg("vulkan_shaders.bin");
    for (shaders.cpps) |cpp| run.addFileArg(cpp);

    const shaders_mod = b.createModule(.{ .root_source_file = table });
    shaders_mod.addAnonymousImport("vulkan_shaders_blob", .{ .root_source_file = blob });
    mod.addImport("vulkan_shaders", shaders_mod);

    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/ggml/vulkan/c_import.h"),
        .target = mod.resolved_target.?,
        .optimize = mod.optimize.?,
    });
    for (vk.include_dirs) |dir| translate.addAfterIncludePath(.{ .cwd_relative = dir });
    mod.addImport("vulkan_c", translate.createModule());
}

/// The functions upstream's `ggml-vulkan.o` exports unmangled, measured with
/// `nm -g --defined-only` (`NOTES.md`, "Stage 7: the measurement"):
/// `ggml-vulkan.h`'s eight and `vk_conv_block_sizes`, an internal table
/// missing its `static`.
pub const vulkan_c_exports = [_][]const u8{
    "ggml_backend_is_vk",
    "ggml_backend_vk_buffer_type",
    "ggml_backend_vk_get_device_count",
    "ggml_backend_vk_get_device_description",
    "ggml_backend_vk_get_device_memory",
    "ggml_backend_vk_host_buffer_type",
    "ggml_backend_vk_init",
    "ggml_backend_vk_reg",
    "vk_conv_block_sizes",
};

/// Builds upstream's Vulkan backend a second time, with `vulkan_c_exports`
/// renamed to `ref_*`, as `libggml-vulkan-ref.a`, for the in-process gates
/// that register it beside the port -- `addCudaRef`'s counterpart.
///
/// It carries the shader `.cpp` objects from the build's one generation, so
/// the reference and the port's table hold the same bytes. Their `*_data`
/// symbols are the C++'s; the port's table defines no symbol at all, so
/// nothing collides.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target and optimize mode; `opts.vulkan` must be set.
///
/// Return: the library; propagates directory errors.
pub fn addVulkanRef(b: *std.Build, opts: Options) !*std.Build.Step.Compile {
    const vk = opts.vulkan.?;
    const shaders = try vulkanShaders(b, vk);
    const mod = b.createModule(.{
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        .sanitize_c = .off,
    });
    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/src"));
    mod.addIncludePath(shaders.header_dir);
    var renames: std.ArrayList([]const u8) = .empty;
    for (vulkan_c_exports) |name| try renames.append(b.allocator, b.fmt("-D{s}=ref_{s}", .{ name, name }));
    for (shaders.cpps) |cpp| mod.addCSourceFile(.{ .file = cpp, .flags = &cxx_flags });
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &.{"ggml/src/ggml-vulkan/ggml-vulkan.cpp"},
        .flags = vulkanCxxFlags(b, vk, renames.items),
    });
    return b.addLibrary(.{ .name = "ggml-vulkan-ref", .root_module = mod, .linkage = .static });
}

/// Compiles `harness/vk_push_layout.cpp`, the `static_assert`s
/// `scripts/gen-vk-push` writes against the push-constant structs of the
/// real `ggml-vulkan.cpp`, which it includes. Compiling it is the check.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target, optimisation and the Vulkan toolchain.
///
/// Return: the object; depending on its step runs the check.
pub fn addVulkanPushLayout(b: *std.Build, opts: Options) !*std.Build.Step.Compile {
    const vk = opts.vulkan.?;
    const shaders = try vulkanShaders(b, vk);
    const mod = b.createModule(.{
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        .link_libcpp = true,
        .sanitize_c = .off,
    });
    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/src"));
    mod.addIncludePath(shaders.header_dir);
    mod.addCSourceFile(.{ .file = b.path("harness/vk_push_layout.cpp"), .flags = vulkanCxxFlags(b, vk, &.{}) });
    return b.addObject(.{ .name = "vk_push_layout", .root_module = mod });
}

/// Adds the Vulkan loader to a module's link: every module that links a
/// `libggml.a` built with `-Dvulkan` calls this, as `addCudaLibraries`.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the module whose link gains the loader.
/// - `vulkan`: the toolchain; null adds nothing.
pub fn addVulkanLibraries(_: *std.Build, mod: *std.Build.Module, vulkan: ?Vulkan) void {
    const vk = vulkan orelse return;
    for (vk.lib_dirs) |dir| mod.addLibraryPath(.{ .cwd_relative = dir });
    mod.linkSystemLibrary("vulkan", .{});
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
            .optimize = .debug,
        }),
    });

    const run = b.addRunArtifact(tool);
    run.addArg(kind);
    const out_metal = run.addOutputFileArg2(b.fmt("ggml-metal-embed-{s}.metal", .{kind}), .{});
    const out_asm = run.addOutputFileArg2(b.fmt("ggml-metal-embed-{s}.s", .{kind}), .{});
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
    // ggml-threading.cpp used to be here for the critical section; it is
    // ported now (src/ggml/threading.zig) and comes from `ported.zig`.
    //
    // Linkable since the graph section landed: it needs the graph and
    // allocator symbols, which the port now provides. Before that,
    // `ported.zig` stubbed the one backend function the ported code called.
    // ggml-backend.cpp used to be here; it is ported now
    // (src/ggml/backend.zig, src/ggml/backend_sched.zig) and comes from
    // `ported.zig`, and so does the meta-buffer backend it calls into
    // (src/ggml/backend_meta.zig).
    // All that is left under ggml-cpu/ are three translation units that are
    // empty on this target: hbm.cpp needs GGML_USE_CPU_HBM, the two amx
    // files __AMX_INT8__.
    "ggml/src/ggml-cpu/hbm.cpp",
    "ggml/src/ggml-cpu/amx/amx.cpp",
    "ggml/src/ggml-cpu/amx/mmq.cpp",
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
    // No `-DGGML_USE_METAL` in this archive's define list and no Metal in its
    // link, exactly as in the ported test root above. The ported registry
    // must not reference the Metal backend.
    options.addOption(bool, "use_metal", false);
    options.addOption(bool, "use_cuda", false);
    options.addOption(bool, "use_vulkan", false);
    options.addOption(bool, "vulkan_port", false);
    options.addOption(bool, "use_blas", false);
    options.addOption(bool, "blas_port", false);
    options.addOption(bool, "cuda_port", false);
    mod.addOptions("config", options);
    mod.addImport("ggml_c", ggmlCModule(b, opts));
    addVarargsShim(b, mod, opts);

    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/src"));
    // The CPU sources reach their own headers unqualified.
    mod.addIncludePath(b.path(root ++ "/ggml/src/ggml-cpu"));

    const common_defines = [_][]const u8{
        "-DGGML_USE_CPU",
        "-DGGML_SCHED_MAX_COPIES=4",
        "-DGGML_USE_LLAMAFILE",
        // Upstream's CMake defaults GGML_CPU_REPACK to ON
        // (ggml/CMakeLists.txt:152). Without it `ggml-cpu.cpp` never
        // registers the repack buffer type, the extra-buffer list is
        // empty, and `repack.cpp` plus `arch/arm/repack.cpp` compile
        // into the library unreachable -- 5,692 live lines of dead
        // code. Measured, by panicking on the one entry point.
        "-DGGML_USE_CPU_REPACK",
    };
    const defines = flags(b, &.{ &common_defines, if (isApple(opts)) &accelerate_defines else &.{} });

    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ported_test_c_sources,
        .flags = flags(b, &.{ &c_flags, defines }),
    });
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ported_test_cxx_sources,
        .flags = flags(b, &.{ &cxx_flags, defines }),
    });

    if (isApple(opts)) {
        addSdkPaths(b, mod, opts.sdk);
        // `-DGGML_USE_ACCELERATE` sends the vector helpers in `ops.cpp` and
        // `binary-ops.cpp` to vDSP, so the framework has to come with them.
        mod.linkFramework("Accelerate", .{});
    }
    // LLVM's backend, for the intrinsics; see `addGgml`.
    const lib = b.addLibrary(.{ .name = "ggml-ported", .root_module = mod, .linkage = .static, .use_llvm = true });
    lib.setLibCFile(opts.libc_file);
    return lib;
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
    // No `-DGGML_USE_METAL` in this root's define list, and no Metal sources
    // in its link. The ported registry must not reference the Metal backend.
    options.addOption(bool, "use_metal", false);
    options.addOption(bool, "use_cuda", false);
    options.addOption(bool, "use_vulkan", false);
    options.addOption(bool, "vulkan_port", false);
    options.addOption(bool, "use_blas", false);
    options.addOption(bool, "blas_port", false);
    options.addOption(bool, "cuda_port", false);
    mod.addOptions("config", options);
    mod.addImport("ggml_c", ggmlCModule(b, opts));
    addVarargsShim(b, mod, opts);

    mod.addIncludePath(b.path(root ++ "/ggml/include"));
    mod.addIncludePath(b.path(root ++ "/ggml/src"));
    // The CPU sources reach their own headers unqualified.
    mod.addIncludePath(b.path(root ++ "/ggml/src/ggml-cpu"));

    const common_defines = [_][]const u8{
        "-DGGML_USE_CPU",
        "-DGGML_SCHED_MAX_COPIES=4",
        "-DGGML_USE_LLAMAFILE",
        // Upstream's CMake defaults GGML_CPU_REPACK to ON
        // (ggml/CMakeLists.txt:152). Without it `ggml-cpu.cpp` never
        // registers the repack buffer type, the extra-buffer list is
        // empty, and `repack.cpp` plus `arch/arm/repack.cpp` compile
        // into the library unreachable -- 5,692 live lines of dead
        // code. Measured, by panicking on the one entry point.
        "-DGGML_USE_CPU_REPACK",
    };
    const defines = flags(b, &.{ &common_defines, if (isApple(opts)) &accelerate_defines else &.{} });

    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = c_sources,
        .flags = flags(b, &.{ &c_flags, defines }),
    });
    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = &ported_test_cxx_sources,
        .flags = flags(b, &.{ &cxx_flags, defines }),
    });

    if (isApple(opts)) {
        addSdkPaths(b, mod, opts.sdk);
        mod.linkFramework("Accelerate", .{});
    }
    return mod;
}

/// Builds libllama: the port, rooted at `src/llama/module.zig`, whose
/// `exports.zig` exports every function of `llama.h`. No C or C++ compiles
/// into it.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target, optimize mode, and SDK.
/// - `ggml`: the ggml archive it links against.
///
/// Return: the library.
fn addLlama(b: *std.Build, opts: Options, ggml: *std.Build.Step.Compile) *std.Build.Step.Compile {
    const mod = b.createModule(.{
        .root_source_file = b.path("src/llama/module.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
    });
    mod.addImport("llama_c", llamaCModule(b, opts));
    addSdkPaths(b, mod, opts.sdk);
    mod.linkLibrary(ggml);
    // Where `libggml` links the Vulkan loader -- its unit tests do -- Zig
    // resolves it again for every library that links `libggml`, and the
    // search path does not travel with it.
    if (opts.vulkan) |vk| for (vk.lib_dirs) |lib_dir| mod.addLibraryPath(.{ .cwd_relative = lib_dir });
    addCudaLibraries(b, mod, opts.cuda);

    const lib = b.addLibrary(.{ .name = "llama", .root_module = mod, .linkage = .static });
    lib.setLibCFile(opts.libc_file);
    return lib;
}

/// Builds upstream's libllama, every translation unit and every
/// architecture under `src/models/`, as `libllama-ref.a`: the reference the
/// in-process harnesses link beside the port.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target, optimize mode, and SDK.
/// - `ggml`: the ggml archive it links against.
///
/// Return: the library; propagates errors from reading `src/models/`.
fn addLlamaRef(b: *std.Build, opts: Options, ggml: *std.Build.Step.Compile) !*std.Build.Step.Compile {
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
    try files.appendSlice(b.allocator, &llama_ref_sources);

    // Configure is cached, and this listing is not something Zig can see on
    // its own: a model added or removed upstream must configure again.
    b.dependOnDirectoryMetadata(b.path(root ++ "/src/models"));
    var dir = try b.root.openDir(b.graph.io, root ++ "/src/models", .{ .iterate = true });
    defer dir.close(b.graph.io);
    var it = dir.iterate();
    while (try it.next(b.graph.io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".cpp")) continue;
        try files.append(b.allocator, b.fmt("src/models/{s}", .{entry.name}));
    }

    // src/CMakeLists.txt:55, configure_file(llama-version.h.in).
    mod.addConfigHeader(b.addConfigHeader(.{
        .style = .{ .cmake = b.path(root ++ "/src/llama-version.h.in") },
        .include_path = "llama-version.h",
    }, .{ .LLAMA_VERSION = upstream.version, .LLAMA_BUILD_COMMIT = upstream.commit }));

    const common_defines = [_][]const u8{"-DGGML_USE_CPU"};
    const apple_defines = [_][]const u8{"-DGGML_USE_METAL"};

    mod.addCSourceFiles(.{
        .root = b.path(root),
        .files = files.items,
        .flags = flags(b, &.{ &cxx_flags, &common_defines, if (isApple(opts)) &apple_defines else &.{} }),
    });

    addSdkPaths(b, mod, opts.sdk);
    mod.linkLibrary(ggml);
    // Where `libggml` links the Vulkan loader -- its unit tests do -- Zig
    // resolves it again for every library that links `libggml`, and the
    // search path does not travel with it.
    if (opts.vulkan) |vk| for (vk.lib_dirs) |lib_dir| mod.addLibraryPath(.{ .cwd_relative = lib_dir });
    addCudaLibraries(b, mod, opts.cuda);

    const lib = b.addLibrary(.{ .name = "llama-ref", .root_module = mod, .linkage = .static });
    lib.setLibCFile(opts.libc_file);
    return lib;
}
