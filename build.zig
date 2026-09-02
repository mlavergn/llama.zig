const std = @import("std");
const builtin = @import("builtin");

const zon = @import("build.zig.zon");

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
    // Build config
    const cfg = Config{
        .name = @tagName(zon.name),
        .mod_name = @tagName(zon.name),
        .cli_name = @tagName(zon.name) ++ "-cli",
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
    // CLI

    const cli_module = b.createModule(.{
        .root_source_file = cfg.cli_source_file,
        .target = cfg.target,
        .optimize = cfg.optimize,
    });

    const cli = b.addExecutable(.{
        .name = cfg.name,
        .root_module = cli_module,
    });
    cli.root_module.addOptions("config", options);
    cli.root_module.addImport(cfg.mod_name, module);

    cli.root_module.addIncludePath(b.path("src"));
    cli.root_module.linkLibrary(lib);

    // depend on lib being built
    cli.step.dependOn(&lib.step);
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
    });
    cli_tests_module.addOptions("config", options);
    cli_tests_module.addImport(cfg.mod_name, module);

    const cli_tests = b.addTest(.{
        .root_module = cli_tests_module,
    });

    const cli_tests_run = b.addRunArtifact(cli_tests);
    cli_tests_run.has_side_effects = true;
    tests_step.dependOn(&cli_tests_run.step);

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
