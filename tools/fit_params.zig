//! `llama-fit-params`: print the command-line arguments `--fit` would choose,
//! or the memory a model would take.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/fit-params/fit-params.cpp` at v0.3.0
//! (`c1d0e7a00`). The fitting itself is `cli/fit.zig`, the port of
//! `common/fit.cpp` gated by `make fit-diff`; this is the command line
//! around it.

const std = @import("std");
const tc = @import("tools_common");
const fit = @import("cli_fit");
const c = tc.c;
const io = tc.io;
const log = tc.log;
const arg = tc.arg;

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

/// Ports `llama_fit_params` (fit-params.cpp:18 @c1d0e7a00).
fn run(init: std.process.Init, a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    const func = "llama_fit_params";
    var params = try tc.params.Params.init(a);

    tc.common.init();

    if (!try arg.parse(argv, &params, .fit_params, null)) return 1;

    c.llama_backend_init();
    c.llama_numa_init(params.numa);

    var mparams = tc.common.modelParamsToLlama(&params);
    var cparams = tc.common.contextParamsToLlama(&params);

    const fit_log: fit.Log = .{ .verbosity = std.math.maxInt(i32), .sink = tc.common.fitLogSink };
    const model_path = try a.dupeSentinel(u8, params.model.path, 0);

    if (!params.fit_params_print) {
        var patterns = try fit.Patterns.init(a);
        const status = try fit.fitParams(
            init.gpa,
            &patterns,
            fit_log,
            model_path.ptr,
            @ptrCast(&mparams),
            @ptrCast(&cparams),
            &params.tensor_split,
            @ptrCast(params.tensor_buft_overrides.items.ptr),
            params.fit_params_target.items.ptr,
            @bitCast(params.fit_params_min_ctx),
            null,
            if (params.verbosity >= log.level_debug) c.GGML_LOG_LEVEL_DEBUG else c.GGML_LOG_LEVEL_ERROR,
        );
        if (status != .success) {
            log.err("%s: failed to fit CLI arguments to free memory, exiting...\n", .{func});
            return 1;
        }

        log.inf("%s: printing fitted CLI arguments to stdout...\n", .{func});
        io.printf("-c %u -ngl %d", .{ cparams.n_ctx, mparams.n_gpu_layers });

        var nd = c.llama_max_devices();
        while (nd > 1 and mparams.tensor_split[nd - 1] == 0.0) nd -= 1;
        if (nd > 1) {
            var id: usize = 0;
            while (id < nd) : (id += 1) {
                if (id == 0) io.printf(" -ts ", .{});
                io.printf("%s%u", .{ @as([*:0]const u8, if (id > 0) "," else ""), @as(u32, @intFromFloat(mparams.tensor_split[id])) });
            }
        }

        const ntbo = c.llama_max_tensor_buft_overrides();
        var any_tbo = false;
        var itbo: usize = 0;
        while (itbo < ntbo and mparams.tensor_buft_overrides[itbo].pattern != null) : (itbo += 1) {
            if (itbo == 0) io.printf(" -ot \"", .{});
            io.printf("%s%s=%s", .{
                @as([*:0]const u8, if (itbo > 0) "," else ""),
                mparams.tensor_buft_overrides[itbo].pattern,
                c.ggml_backend_buft_name(mparams.tensor_buft_overrides[itbo].buft),
            });
            any_tbo = true;
        }
        io.printf("%s\n", .{@as([*:0]const u8, if (any_tbo) "\"" else "")});
    } else {
        log.inf("%s: printing estimated memory in MiB to stdout (device, model, context, compute) ...\n", .{func});

        io.flushStdout();
        var out_buf: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
        fit.fitPrint(init.gpa, fit_log, &stdout.interface, model_path.ptr, @ptrCast(&mparams), @ptrCast(&cparams)) catch |e| switch (e) {
            // the C's exception escapes `main`: libc++ reports it and aborts
            error.Runtime => {
                stdout.interface.flush() catch {};
                io.eprintf("libc++abi: terminating due to uncaught exception of type std::runtime_error: %s\n", .{@as([*:0]const u8, &fit.print_what)});
                std.process.abort();
            },
            else => |other| return other,
        };
        try stdout.interface.flush();
    }

    return 0;
}

/// Ports `main` (tools/fit-params/main.cpp:3 @c1d0e7a00).
pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(init, arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
