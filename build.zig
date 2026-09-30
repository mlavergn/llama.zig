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
        inline for (@typeInfo(@TypeOf(zon.dependencies)).@"struct".fields) |field| {
            if (comptime Self.isCloneable(field.name) and !Self.isCloned(field.name)) {
                const dep = @field(zon.dependencies, field.name);
                const dest = b.pathFromRoot(dep.path);
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
        inline for (@typeInfo(@TypeOf(zon.dependencies)).@"struct".fields) |field| {
            if (Self.isCloneable(field.name) and !Self.isCloned(field.name)) return false;
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

    /// Resolves the Apple Silicon macOS SDK and populates the handle.
    ///
    /// Runs during the build on macOS so the library and CLI can be given the
    /// framework search path they need to link against system frameworks.
    /// Must be called exactly once after `init` and before `target` or `sdk`
    /// are read; on success both fields are valid for the handle's lifetime.
    ///
    /// Parameters:
    /// - `self`: the handle to populate; mutated in place.
    ///
    /// Return: nothing on success; returns `error.FailedToResolveSDK` when no
    /// SDK can be located, and propagates target-resolution errors.
    pub fn resolve(self: *Self) !void {
        const query = std.Target.Query{
            .cpu_arch = .aarch64,
            .os_tag = .macos,
        };
        self.target = try std.zig.system.resolveTargetQuery(self.io, query);
        self.sdk = std.zig.system.darwin.getSdk(
            self.allocator,
            self.io,
            &self.target,
        ) orelse return error.FailedToResolveSDK;
    }
};

const Config = struct {
    name: []const u8,
    mod_name: []const u8,
    cli_name: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
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
pub fn build(b: *std.Build) !void {
    // Pre-flight: ensure any local dependencies are cloned
    Git.cloneDeps(b);

    // Build config
    const cfg = Config{
        .name = @tagName(zon.name),
        .mod_name = @tagName(zon.name),
        .cli_name = "llama-cli",
        .target = b.standardTargetOptions(.{}),
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
        try xcode.resolve();
        lib.root_module.addFrameworkPath(.{ .cwd_relative = xcode.sdk });
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
        try xcode.resolve();
        break :blk xcode.sdk;
    } else null;

    // Proves ported code is on the execution path. See scripts/probe-ported.
    const probe_ported = b.option(bool, "probe-ported", "Abort from ported code on a hot path, to prove it runs") orelse false;
    const probe_cpu = b.option(bool, "probe-cpu", "Abort from the ported CPU backend, to prove it runs") orelse false;

    const reference = try llamacpp.add(b, .{
        .target = cfg.target,
        .optimize = cfg.optimize,
        .sdk = sdk,
        .probe_ported = probe_ported,
        .probe_cpu = probe_cpu,
    });

    const ggml_install = b.addInstallArtifact(reference.ggml, .{});
    const llama_install = b.addInstallArtifact(reference.llama, .{});

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

    const reference_step = b.step("reference", "Build the llama.cpp reference libraries");
    reference_step.dependOn(&ggml_install.step);
    reference_step.dependOn(&llama_install.step);

    // End-to-end check: load a model through libllama's C ABI and generate.
    // Compiling and linking says nothing about whether inference works, and
    // the Metal path in particular can only be proven by running it.

    const smoke_module = b.createModule(.{
        .root_source_file = b.path("harness/smoke.zig"),
        .target = cfg.target,
        .optimize = cfg.optimize,
        .link_libc = true,
        // Apple's libc++, not Zig's -- see the note in build/llamacpp.zig.
        .link_libcpp = false,
    });
    smoke_module.addIncludePath(b.path("llama.cpp/include"));
    smoke_module.addIncludePath(b.path("llama.cpp/ggml/include"));
    smoke_module.linkLibrary(reference.llama);
    smoke_module.linkLibrary(reference.ggml);
    if (sdk) |path| smoke_module.addFrameworkPath(.{ .cwd_relative = path });
    llamacpp.linkAppleLibcxx(b, smoke_module, sdk);

    const smoke = b.addExecutable(.{ .name = "smoke", .root_module = smoke_module });

    const smoke_run = b.addRunArtifact(smoke);
    smoke_run.has_side_effects = true;
    if (b.args) |args| smoke_run.addArgs(args);

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
        // Apple's libc++, not Zig's -- see the note in build/llamacpp.zig.
        .link_libcpp = false,
    });
    cli_module.addOptions("config", options);
    // The one permitted dependency, and only here: chat templates are Jinja,
    // and `libllamazig` must stay dependency-free. See PLAN.md Decisions 25
    // and 27.
    cli_module.addImport("zigjinja", jinja_module);
    cli_module.addIncludePath(b.path("llama.cpp/include"));
    cli_module.addIncludePath(b.path("llama.cpp/ggml/include"));
    cli_module.linkLibrary(reference.llama);
    cli_module.linkLibrary(reference.ggml);
    if (sdk) |path| cli_module.addFrameworkPath(.{ .cwd_relative = path });
    llamacpp.linkAppleLibcxx(b, cli_module, sdk);

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
    if (b.args) |args| {
        cli_run.addArgs(args);
    }

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
        // Apple's libc++, not Zig's -- see the note in build/llamacpp.zig.
        .link_libcpp = false,
    });
    cli_tests_module.addOptions("config", options);
    cli_tests_module.addImport("zigjinja", jinja_module);
    // `session.zig` imports llama.h, so even the argument-parsing tests need
    // the header and the library behind it.
    cli_tests_module.addIncludePath(b.path("llama.cpp/include"));
    cli_tests_module.addIncludePath(b.path("llama.cpp/ggml/include"));
    cli_tests_module.linkLibrary(reference.llama);
    cli_tests_module.linkLibrary(reference.ggml);
    if (sdk) |path| cli_tests_module.addFrameworkPath(.{ .cwd_relative = path });
    llamacpp.linkAppleLibcxx(b, cli_tests_module, sdk);

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
        try docs_xcode.resolve();
        docs_lib.root_module.addFrameworkPath(.{ .cwd_relative = docs_xcode.sdk });
    }

    const docs = b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });

    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&docs.step);
}
