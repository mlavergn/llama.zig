//! `llama-fit-params`: print the command-line arguments `--fit` would choose,
//! or the memory a model would take.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/fit-params/fit-params.cpp` at v1.0.0
//! (`b1b770d90`). The fitting itself is `cli/fit.zig`, the port of
//! `common/fit.cpp` gated by `make fit-diff`; this is the command line
//! around it.

const std = @import("std");
const llama = @import("llama");
const stdlib = @import("stdlib");
const tools_common = @import("tools_common");
const cli_fit = @import("cli_fit");

/// Ports `llama_fit_params` (fit-params.cpp:17 @b1b770d90).
fn run(init: std.process.Init, a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    const func = "llama_fit_params";
    var params = try tools_common.params.Params.init(a);

    tools_common.common.init();

    if (!try tools_common.arg.parse(argv, &params, .fit_params, null)) return 1;

    llama.api.llama.llama_backend_init();
    llama.api.llama.llama_numa_init(params.numa);

    var mparams = tools_common.common.modelParamsToLlama(&params);
    var cparams = tools_common.common.contextParamsToLlama(&params);

    const fit_log: cli_fit.Log = .{ .verbosity = std.math.maxInt(i32), .sink = tools_common.common.fitLogSink, .json = tools_common.common.fitLogJson };
    const model_paths = try tools_common.common.modelPaths(a, &params.model);

    if (!params.fit_params_print) {
        var patterns = try cli_fit.Patterns.init(a);
        const status = try cli_fit.fitParams(
            init.gpa,
            &patterns,
            fit_log,
            model_paths,
            @ptrCast(&mparams),
            @ptrCast(&cparams),
            &params.tensor_split,
            @ptrCast(params.tensor_buft_overrides.items.ptr),
            params.fit_params_target.items.ptr,
            @bitCast(params.fit_params_min_ctx),
            null,
            if (params.verbosity >= tools_common.log.level_debug) llama.impl.llama_c.GGML_LOG_LEVEL_DEBUG else llama.impl.llama_c.GGML_LOG_LEVEL_ERROR,
        );
        if (status != .success) {
            tools_common.log.err("%s: failed to fit CLI arguments to free memory, exiting...\n", .{func});
            return 1;
        }

        tools_common.log.inf("%s: printing fitted CLI arguments to stdout...\n", .{func});
        tools_common.io.printf("-c %u -ngl %d", .{ cparams.n_ctx, mparams.n_gpu_layers });

        var nd = llama.api.llama.llama_max_devices();
        while (nd > 1 and mparams.tensor_split[nd - 1] == 0.0) nd -= 1;
        if (nd > 1) {
            var id: usize = 0;
            while (id < nd) : (id += 1) {
                if (id == 0) tools_common.io.printf(" -ts ", .{});
                tools_common.io.printf("%s%u", .{ @as([*:0]const u8, if (id > 0) "," else ""), @as(u32, @intFromFloat(mparams.tensor_split[id])) });
            }
        }

        const ntbo = llama.api.llama.llama_max_tensor_buft_overrides();
        var any_tbo = false;
        var itbo: usize = 0;
        while (itbo < ntbo and mparams.tensor_buft_overrides[itbo].pattern != null) : (itbo += 1) {
            if (itbo == 0) tools_common.io.printf(" -ot \"", .{});
            tools_common.io.printf("%s%s=%s", .{
                @as([*:0]const u8, if (itbo > 0) "," else ""),
                mparams.tensor_buft_overrides[itbo].pattern,
                llama.impl.llama_c.ggml_backend_buft_name(mparams.tensor_buft_overrides[itbo].buft),
            });
            any_tbo = true;
        }
        tools_common.io.printf("%s\n", .{@as([*:0]const u8, if (any_tbo) "\"" else "")});
    } else {
        tools_common.log.inf("%s: printing estimated memory in MiB to stdout (device, model, context, compute) ...\n", .{func});

        stdlib.stdio.stdout().flush();
        var out_buf: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
        cli_fit.fitPrint(init.gpa, fit_log, &stdout.interface, model_paths, @ptrCast(&mparams), @ptrCast(&cparams)) catch |e| switch (e) {
            // the C's exception escapes `main`: libc++ reports it and aborts
            error.Runtime => {
                stdout.interface.flush() catch {};
                tools_common.io.eprintf("libc++abi: terminating due to uncaught exception of type std::runtime_error: %s\n", .{@as([*:0]const u8, &cli_fit.print_what)});
                std.process.abort();
            },
            else => |other| return other,
        };
        try stdout.interface.flush();
    }

    return 0;
}

/// Ports `main` (tools/fit-params/main.cpp:3 @b1b770d90).
pub fn main(init: std.process.Init) !u8 {
    stdlib.stdio.bufferAsC();
    stdlib.env.set(init.environ_map);
    defer stdlib.stdio.flushAll();
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(init, arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
