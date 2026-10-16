const std = @import("std");
const builtin = @import("builtin");

const zon = @import("build.zig.zon");
const llamacpp = @import("build/llamacpp.zig");

const Git = struct {
    const Self = @This();

    /// Clones every missing cloneable dependency and exits 1; returns when none are missing.
    ///
    /// Parameters:
    /// - `b`: the build graph.
    ///
    /// Return: nothing when all are present; otherwise never.
    pub fn cloneDeps(b: *std.Build) if (Self.hasAllDeps()) void else noreturn {
        if (comptime Self.hasAllDeps()) return;
        const io = b.graph.io;
        var missing: usize = 0;
        var cloned: usize = 0;
        inline for (@typeInfo(@TypeOf(zon.dependencies)).@"struct".field_names) |name| {
            if (comptime Self.isCloneable(name) and !Self.isCloned(name)) {
                const dep = @field(zon.dependencies, name);
                const dest = b.root.joinString(b.allocator, dep.path) catch @panic("OOM");
                missing += 1;
                if (std.Io.Dir.cwd().access(io, dest, .{})) |_| {
                    std.debug.print("{s} exists but is not a Zig package\n", .{dep.path});
                } else |err| switch (err) {
                    error.FileNotFound => if (Self.clone(io, dep.clone, dest)) {
                        cloned += 1;
                    } else |clone_err| {
                        std.debug.print("git clone {s} {s} failed [{any}]\n", .{ dep.clone, dep.path, clone_err });
                    },
                    else => std.debug.print("cannot check {s} [{any}]\n", .{ dep.path, err }),
                }
            }
        }
        if (cloned > 0) std.debug.print("cloned {d} of {d} missing dependencies; re-run zig build\n", .{ cloned, missing });
        std.process.exit(1);
    }

    /// Reports whether every cloneable dependency is present.
    ///
    /// Return: `true` when none is missing.
    fn hasAllDeps() bool {
        inline for (@typeInfo(@TypeOf(zon.dependencies)).@"struct".field_names) |name| {
            if (Self.isCloneable(name) and !Self.isCloned(name)) return false;
        }
        return true;
    }

    /// Reports whether dependency `name` was present when the build runner was compiled.
    ///
    /// Parameters:
    /// - `name`: the dependency's field name in `build.zig.zon`.
    ///
    /// Return: `true` when present, or when this package is not the root.
    fn isCloned(comptime name: []const u8) bool {
        const deps = @import("root").dependencies;
        for (deps.root_deps) |dep| {
            if (std.mem.eql(u8, dep[0], name)) return @hasDecl(@field(deps.packages, dep[1]), "build_zig");
        }
        return true;
    }

    /// Reports whether dependency `name` declares both a `.path` and a `.clone`.
    ///
    /// Parameters:
    /// - `name`: the dependency's field name in `build.zig.zon`.
    ///
    /// Return: `true` when both fields are present.
    fn isCloneable(comptime name: []const u8) bool {
        const Dep = @TypeOf(@field(zon.dependencies, name));
        return @hasField(Dep, "path") and @hasField(Dep, "clone");
    }

    /// Clones the repository at `url` into `dest`.
    ///
    /// Parameters:
    /// - `io`: IO the `git` child is spawned on.
    /// - `url`: the repository to clone.
    /// - `dest`: the directory to clone into.
    ///
    /// Return: nothing on success; `error.GitCloneFailed` when `git` fails.
    fn clone(io: std.Io, url: []const u8, dest: []const u8) !void {
        var child = try std.process.spawn(io, .{ .argv = &.{ "git", "clone", url, dest } });
        switch (try child.wait(io)) {
            .exited => |code| if (code != 0) return error.GitCloneFailed,
            else => return error.GitCloneFailed,
        }
    }
};

const Xcode = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    io: std.Io,
    target: std.Target,
    sdk: []const u8,

    /// Constructs an `Xcode` handle ready to resolve the macOS SDK.
    ///
    /// Bundles the allocator and IO the SDK lookup needs so callers pass a
    /// single value rather than threading both through `resolve`. The returned
    /// handle is inert until `resolve` runs; its target and SDK path are unset
    /// and must not be read before then.
    ///
    /// Parameters:
    /// - `allocator`: allocator used by the later SDK resolution; must outlive the handle.
    /// - `io`: IO interface used to query the host system during resolution.
    ///
    /// Return: an initialized `Xcode` handle; the error union is reserved for
    /// the lifecycle convention and yields no error today.
    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Self {
        return Self{
            .allocator = allocator,
            .io = io,
            // SAFETY: populated by resolve() before either field is read.
            .target = undefined,
            // SAFETY: populated by resolve() before either field is read.
            .sdk = undefined,
        };
    }

    /// Resolves the Apple SDK for the build target and populates the handle.
    ///
    /// Runs during the build on macOS so the library and CLI can be given the
    /// framework search path they need to link against system frameworks.
    /// An Apple target gets its own SDK -- macOS, iPhoneOS or
    /// iPhoneSimulator, which is what lets `xcfwk/` build the iOS slices.
    /// Anything else falls back to the Apple Silicon macOS SDK, as before.
    /// Must be called exactly once after `init` and before `target` or `sdk`
    /// are read; on success both fields are valid for the handle's lifetime.
    ///
    /// Zig 0.17 reuses a configured build graph until something it was told
    /// about changes, and the SDK lookup is not something it can see. The
    /// SDK's path carries its version (`MacOSX27.0.sdk`), so its
    /// `SDKSettings.json` is made a configure input: an Xcode update removes
    /// the path, and the next build configures again.
    ///
    /// Parameters:
    /// - `self`: the handle to populate; mutated in place.
    /// - `b`: the build graph, which records the SDK as a configure input.
    /// - `target`: the build target; selects which SDK.
    ///
    /// Return: nothing on success; returns `error.FailedToResolveSDK` when no
    /// SDK can be located, and propagates target-resolution errors.
    pub fn resolve(self: *Self, b: *std.Build, target: std.Target) !void {
        const query = std.Target.Query{
            .cpu_arch = .aarch64,
            .os_tag = .macos,
        };
        self.target = if (target.os.tag.isDarwin())
            target
        else
            try std.zig.system.resolveTargetQuery(self.io, query);
        self.sdk = std.zig.system.darwin.getSdk(
            self.allocator,
            self.io,
            &self.target,
        ) orelse return error.FailedToResolveSDK;
        b.dependOnFileMetadata(b.graph.cwdRelativePath(b.pathJoin(&.{ self.sdk, "SDKSettings.json" })));
    }
};

const Config = struct {
    name: []const u8,
    mod_name: []const u8,
    cli_name: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.lang.Optimize,
    module_source_file: std.Build.LazyPath,
    cli_source_file: std.Build.LazyPath,
    version: std.SemanticVersion,
};

/// Declares the build graph for the llamazig library, CLI, tests, and docs.
///
/// Serves as the entry point the Zig build runner invokes to wire up every
/// build step consumers rely on — `lib`, `cli`, `run`, `test`, and `docs` —
/// plus the standard target, optimize, and test-filter options. On macOS it
/// adds the resolved SDK framework path so linking succeeds. Called once per
/// build invocation.
///
/// Parameters:
/// - `b`: the build graph the steps and options are registered on.
///
/// Return: nothing on success; propagates errors from option parsing, version
/// parsing, and macOS SDK resolution.
/// Resolves the `-Dcuda`, `-Dcuda-toolkit` and `-Dcuda-arch` options.
///
/// CUDA defaults to on for a Linux target when `<toolkit>/bin/nvcc` exists,
/// so a plain `zig build cli` on a CUDA machine uses the GPU; `-Dcuda=false`
/// turns it off. The architecture defaults to what `nvidia-smi` reports for
/// the first GPU, with a Blackwell `12X` rewritten to `12Xa` exactly as
/// upstream's CMake does (ggml/src/ggml-cuda/CMakeLists.txt).
///
/// Parameters:
/// - `b`: the build graph the options are registered on.
/// - `target`: the resolved target; CUDA is only defaulted on for Linux.
///
/// Return: the toolkit and architecture, or null when CUDA is off.
fn cudaOptions(b: *std.Build, target: std.Build.ResolvedTarget) ?llamacpp.Cuda {
    const toolkit = b.option([]const u8, "cuda-toolkit", "CUDA toolkit root (default /usr/local/cuda)") orelse "/usr/local/cuda";
    const nvcc = b.pathJoin(&.{ toolkit, "bin", "nvcc" });
    const found = if (std.Io.Dir.accessAbsolute(b.graph.io, nvcc, .{})) true else |_| false;
    const default_on = target.result.os.tag == .linux and found;
    const enabled = b.option(bool, "cuda", "Build the CUDA backend (default: on for Linux when nvcc is found)") orelse default_on;
    if (!enabled) return null;

    const arch = b.option([]const u8, "cuda-arch", "GPU architecture to compile for, e.g. 121a (default: from nvidia-smi)") orelse detect: {
        var code: u8 = 0;
        const out = b.runAllowFail(&.{ "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader" }, &code, .ignore) catch
            std.debug.panic("-Dcuda: nvidia-smi found no GPU; pass -Dcuda-arch=<e.g. 121a>", .{});
        const line = std.mem.trim(u8, std.mem.sliceTo(out, '\n'), " \r\t");
        var digits: std.ArrayList(u8) = .empty;
        for (line) |ch| if (ch != '.') digits.append(b.allocator, ch) catch @panic("OOM");
        // 12X is forward-compatible, 12Xa is not, and the FP4 tensor-core
        // instructions need 12Xa.
        if (digits.items.len == 3 and std.mem.startsWith(u8, digits.items, "12")) digits.append(b.allocator, 'a') catch @panic("OOM");
        break :detect digits.items;
    };
    // nvcc compiles host code with the system g++; ask it where its runtime is.
    const libstdcxx = std.mem.trim(u8, b.run(&.{ "g++", "-print-file-name=libstdc++.so" }), " \r\n\t");
    // The swap (PLAN.md, Decision 48): the Zig host layer is the CUDA
    // backend. `-Dcuda-port=false` builds upstream's instead, which only the
    // gates want -- `scripts/cuda-node-diff` diffs the two.
    const port = b.option(bool, "cuda-port", "Use the Zig CUDA host layer (default: true; false builds upstream's, for the gates)") orelse true;
    return .{ .toolkit = toolkit, .arch = arch, .libstdcxx = libstdcxx, .port = port };
}

pub fn build(b: *std.Build) !void {
    // Pre-flight: ensure any local dependencies are cloned
    Git.cloneDeps(b);

    // Build config
    const cfg = Config{
        .name = @tagName(zon.name),
        .mod_name = @tagName(zon.name),
        .cli_name = "llama-cli",
        // Zig 0.17 recognises the M5 as `apple_m5`, with i8mm and SME, where
        // 0.16 fell back to the macOS baseline (`apple_m1`). The ported ARM
        // kernels are the non-i8mm arms only -- see `has_matmul_int8` in
        // src/ggml/cpu/traits.zig -- so the default stays at the baseline the
        // port was built and gated on. `-Dcpu` overrides it.
        .target = b.standardTargetOptions(.{ .default_target = .{ .cpu_model = .baseline } }),
        .optimize = b.standardOptimizeOption(.{}),
        .module_source_file = b.path("src/module.zig"),
        .cli_source_file = b.path("cli/main.zig"),
        .version = try std.SemanticVersion.parse(zon.version),
    };

    // Build options
    const options = b.addOptions();

    // const app_bundle_id = b.option([]const u8, "appBundleId", "Set the appBundleId to use for sandbox contexts") orelse "ai.cloneable";
    // options.addOption([]const u8, "appBundleId", app_bundle_id);
    // cfg.app_bundle_id = app_bundle_id;

    const test_filter = b.option([]const u8, "test-filter", "Run unit tests that match filter") orelse "";
    options.addOption([]const u8, "test_filter", test_filter);

    // `addTest` takes a list, and an empty string would match nothing rather
    // than everything, so an unset option has to become an empty list.
    const test_filters: []const []const u8 =
        if (test_filter.len == 0) &.{} else &.{test_filter};

    // -------------------------------------------------------------------------
    // Module

    const module = b.addModule(cfg.name, .{
        .root_source_file = cfg.module_source_file,
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    module.addOptions("config", options);
    module.addIncludePath(b.path("src"));

    // -------------------------------------------------------------------------
    // Lib

    const lib = b.addLibrary(.{ .name = cfg.name, .root_module = module, .linkage = .static });
    lib.root_module.addOptions("config", options);
    lib.root_module.addIncludePath(b.path("src"));

    if (builtin.os.tag == .macos) {
        var xcode = try Xcode.init(b.allocator, b.graph.io);
        try xcode.resolve(b, cfg.target.result);
        llamacpp.addSdkPaths(b, lib.root_module, xcode.sdk);
    }

    const lib_install = b.addInstallArtifact(lib, .{});
    const lib_step = b.step("lib", "Build static library");
    lib_step.dependOn(&lib_install.step);

    // -------------------------------------------------------------------------
    // llama.cpp reference build
    //
    // Builds the vendored tree with the Zig toolchain, replacing its CMake.
    // These are reference artifacts, not the port: nothing under `llama.cpp/`
    // is modified, and the sources compile exactly as upstream ships them.

    const sdk: ?[]const u8 = if (builtin.os.tag == .macos) blk: {
        var xcode = try Xcode.init(b.allocator, b.graph.io);
        try xcode.resolve(b, cfg.target.result);
        break :blk xcode.sdk;
    } else null;

    // A named Apple `-Dtarget` -- every `xcfwk/` slice -- is a cross-compile
    // to Zig and needs the SDK described to it. A native build does not, and
    // gets nothing new. See `llamacpp.writeAppleLibcFile`.
    const libc_file: ?std.Build.LazyPath =
        if (sdk != null and cfg.target.result.os.tag.isDarwin() and !cfg.target.query.isNative())
            llamacpp.writeAppleLibcFile(b, sdk.?)
        else
            null;

    // Proves ported code is on the execution path. See scripts/probe-ported.
    const probe_ported = b.option(bool, "probe-ported", "Abort from ported code on a hot path, to prove it runs") orelse false;
    const probe_cpu = b.option(bool, "probe-cpu", "Abort from the ported CPU backend, to prove it runs") orelse false;

    const cuda = cudaOptions(b, cfg.target);
    const reference = try llamacpp.add(b, .{
        .target = cfg.target,
        .optimize = cfg.optimize,
        .sdk = sdk,
        .libc_file = libc_file,
        .probe_ported = probe_ported,
        .probe_cpu = probe_cpu,
        .cuda = cuda,
    });

    const ggml_install = b.addInstallArtifact(reference.ggml, .{});
    const llama_install = b.addInstallArtifact(reference.llama, .{});
    const llama_ref_install = b.addInstallArtifact(reference.llama_ref, .{});

    // Tests for ported ggml code that has not been swapped into the library
    // yet. See src/ggml/ported.zig for why this cannot live in `test`.
    const ported_tests = b.addTest(.{
        .root_module = llamacpp.portedTestModule(b, .{
            .target = cfg.target,
            .optimize = cfg.optimize,
            .sdk = sdk,
        }),
        .filters = test_filters,
    });
    const ported_tests_run = b.addRunArtifact(ported_tests);
    ported_tests_run.has_side_effects = true;

    const ported_step = b.step("test-port", "Test ported ggml code not yet swapped in");
    ported_step.dependOn(&ported_tests_run.step);

    // The ported libllama's unit tests. They compile with the library's own
    // module -- rooted at src/llama/module.zig, with the C++ that is still
    // C++ -- because the ported sampler calls into it through llama.h.
    const llama_tests = b.addTest(.{
        .root_module = reference.llama.root_module,
        .filters = test_filters,
    });
    const llama_tests_run = b.addRunArtifact(llama_tests);
    llama_tests_run.has_side_effects = true;
    const llama_tests_step = b.step("test-llama", "Test the ported libllama");
    llama_tests_step.dependOn(&llama_tests_run.step);

    // The ported Zig on its own, so verification can reach it before ggml.c is
    // swapped in. See build/llamacpp.zig.
    const ported_lib = llamacpp.addPortedGgml(b, .{
        .target = cfg.target,
        .optimize = cfg.optimize,
        .sdk = sdk,
        .probe_ported = probe_ported,
        .probe_cpu = probe_cpu,
    });
    const ported_lib_step = b.step("ported-lib", "Build a library from the ported Zig alone");
    ported_lib_step.dependOn(&b.addInstallArtifact(ported_lib, .{}).step);

    // Upstream's CUDA backend with its C exports renamed `ref_*`, for the
    // in-process gates that run it beside the port. See `addCudaRef`.
    if (cuda) |cu| if (cu.port) {
        const cuda_ref = try llamacpp.addCudaRef(b, .{ .target = cfg.target, .optimize = cfg.optimize, .cuda = cuda });
        const cuda_ref_step = b.step("cuda-ref", "Build upstream's CUDA backend with renamed exports");
        cuda_ref_step.dependOn(&b.addInstallArtifact(cuda_ref, .{}).step);
    };

    const reference_step = b.step("reference", "Build the llama.cpp reference libraries");
    reference_step.dependOn(&ggml_install.step);
    reference_step.dependOn(&llama_install.step);
    reference_step.dependOn(&llama_ref_install.step);

    // End-to-end check: load a model through libllama's C ABI and generate.
    // Compiling and linking says nothing about whether inference works, and
    // the Metal path in particular can only be proven by running it.

    // `llama.h`, translated once for the CLI and the smoke harness. Zig 0.17
    // has no `@cImport`; see `cli/c_import.h`.
    const llama_h = b.addTranslateC(.{
        .root_source_file = b.path("cli/c_import.h"),
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    llama_h.addIncludePath(b.path("llama.cpp/include"));
    llama_h.addIncludePath(b.path("llama.cpp/ggml/include"));
    const llama_h_module = llama_h.createModule();

    const smoke_module = b.createModule(.{
        .root_source_file = b.path("harness/smoke.zig"),
        .target = cfg.target,
        .optimize = cfg.optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    smoke_module.addImport("llama_h", llama_h_module);
    smoke_module.linkLibrary(reference.llama);
    smoke_module.linkLibrary(reference.ggml);
    llamacpp.addSdkPaths(b, smoke_module, sdk);
    llamacpp.addCudaLibraries(b, smoke_module, cuda);

    const smoke = b.addExecutable(.{ .name = "smoke", .root_module = smoke_module });

    const smoke_run = b.addRunArtifact(smoke);
    smoke_run.has_side_effects = true;
    smoke_run.addPassthruArgs();

    const smoke_step = b.step("smoke", "Run inference against the reference build");
    smoke_step.dependOn(&smoke_run.step);

    // -------------------------------------------------------------------------
    // CLI
    //
    // Declared after the reference build because it links against it. The CLI
    // is our own Zig binary talking to libllama's C ABI, not upstream's
    // `llama-cli` relinked -- see PLAN.md Decision 20. As `ggml.c` and then
    // llama's own sources are ported, the library underneath changes and this
    // does not.

    // Chat templates are Jinja, so the CLI takes the project's one permitted
    // dependency. `b.dependency` runs its `build.zig`, which is what declares
    // the module.
    const jinja_module = b.dependency("zigjinja", .{
        .target = cfg.target,
        .optimize = cfg.optimize,
    }).module("zigjinja");

    const cli_module = b.createModule(.{
        .root_source_file = cfg.cli_source_file,
        .target = cfg.target,
        .optimize = cfg.optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    cli_module.addOptions("config", options);
    // The one permitted dependency, and only here: chat templates are Jinja,
    // and `libllamazig` must stay dependency-free. See PLAN.md Decisions 25
    // and 27.
    cli_module.addImport("zigjinja", jinja_module);
    cli_module.addImport("llama_h", llama_h_module);
    cli_module.linkLibrary(reference.llama);
    cli_module.linkLibrary(reference.ggml);
    llamacpp.addSdkPaths(b, cli_module, sdk);
    llamacpp.addCudaLibraries(b, cli_module, cuda);

    const cli = b.addExecutable(.{
        .name = cfg.cli_name,
        .root_module = cli_module,
    });
    b.installArtifact(cli);

    const cli_install = b.addInstallArtifact(cli, .{});
    const cli_step = b.step("cli", "Build the CLI app");
    cli_step.dependOn(&cli_install.step);

    // -------------------------------------------------------------------------
    // Run

    const cli_run = b.addRunArtifact(cli);

    // Run step depends on the install step to run from the installation directory
    cli_run.step.dependOn(b.getInstallStep());

    // Support arguments like: `zig build run -- argA argB`
    cli_run.addPassthruArgs();

    const run_step = b.step("run", "Run the CLI app");
    run_step.dependOn(&cli_run.step);

    // -------------------------------------------------------------------------
    // Tests

    const tests = b.addTest(.{
        .root_module = module,
        .filters = test_filters,
    });

    const tests_run = b.addRunArtifact(tests);
    // force tests to run on every test run
    tests_run.has_side_effects = true;

    const tests_step = b.step("test", "Run unit tests");
    tests_step.dependOn(&tests_run.step);

    // The CLI is its own module, so its tests need their own compile step.
    const cli_tests_module = b.createModule(.{
        .root_source_file = b.path("cli/module.zig"),
        .target = cfg.target,
        .optimize = cfg.optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    cli_tests_module.addOptions("config", options);
    cli_tests_module.addImport("zigjinja", jinja_module);
    // `session.zig` imports llama.h, so even the argument-parsing tests need
    // the header and the library behind it.
    cli_tests_module.addImport("llama_h", llama_h_module);
    cli_tests_module.linkLibrary(reference.llama);
    cli_tests_module.linkLibrary(reference.ggml);
    llamacpp.addSdkPaths(b, cli_tests_module, sdk);
    llamacpp.addCudaLibraries(b, cli_tests_module, cuda);

    const cli_tests = b.addTest(.{
        .root_module = cli_tests_module,
        .filters = test_filters,
    });

    const cli_tests_run = b.addRunArtifact(cli_tests);
    cli_tests_run.has_side_effects = true;
    tests_step.dependOn(&cli_tests_run.step);

    // The CLI's tests on their own. `test` also runs the library's, which take
    // about a minute; this step exists so a change under `cli/` can be
    // iterated on -- and fault-injected -- without paying for that.
    const cli_tests_step = b.step("test-cli", "Run the CLI's unit tests only");
    cli_tests_step.dependOn(&cli_tests_run.step);

    // Unit tests for the ported ggml. They compile the whole reference tree,
    // because ported code still calls the parts that have not been ported yet.
    const ggml_tests = b.addTest(.{
        .root_module = reference.ggml_module,
        .filters = test_filters,
    });
    const ggml_tests_run = b.addRunArtifact(ggml_tests);
    ggml_tests_run.has_side_effects = true;
    tests_step.dependOn(&ggml_tests_run.step);

    // -------------------------------------------------------------------------
    // Docs

    // Autodoc roots a module at `root.zig`, or failing that at the file whose
    // basename matches the artifact name; with `src/module.zig` it can do
    // neither, and the site roots itself at an arbitrary import instead.
    // ZIGSTYLE requires the barrel to be `module.zig`, so docs build from a
    // sibling `root.zig`. It is the same module — the other sources come along.
    const docs_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = cfg.target,
        .optimize = cfg.optimize,
    });
    docs_mod.addOptions("config", options);
    docs_mod.addIncludePath(b.path("src"));

    const docs_lib = b.addLibrary(.{
        .name = cfg.name,
        .root_module = docs_mod,
        .linkage = .static,
    });

    if (builtin.os.tag == .macos) {
        var docs_xcode = try Xcode.init(b.allocator, b.graph.io);
        try docs_xcode.resolve(b, cfg.target.result);
        llamacpp.addSdkPaths(b, docs_lib.root_module, docs_xcode.sdk);
    }

    const docs = b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });

    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&docs.step);
}
