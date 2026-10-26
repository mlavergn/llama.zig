//! `llama-batched-bench`: time prompt processing and generation over batch
//! shapes.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/batched-bench/batched-bench.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration names the C++ it replaces and the line it
//! began at. The tokens are the C library's `rand()` from its default seed,
//! as upstream's are, so both decode the same batches.

const std = @import("std");
const tc = @import("tools_common");
const c = tc.c;
const io = tc.io;
const log = tc.log;
const arg = tc.arg;

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

extern fn rand() c_int;

/// Ports `print_usage` (batched-bench.cpp:12 @c1d0e7a00).
fn printUsage(argv: []const [:0]const u8) void {
    log.out("\nexample usage:\n", .{});
    log.out("\n    %s -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 [-pps]\n", .{argv[0].ptr});
    log.out("\n", .{});
}

/// The `decode_helper` lambda (batched-bench.cpp:82 @c1d0e7a00): `batch` in
/// views of `n_batch` tokens.
fn decodeHelper(ctx: ?*c.llama_context, batch: *c.llama_batch, n_batch: i32, synchronize: bool) bool {
    var i: i32 = 0;
    while (i < batch.n_tokens) : (i += n_batch) {
        const n_tokens = @min(n_batch, batch.n_tokens - i);
        const off: usize = @intCast(i);

        const batch_view: c.llama_batch = .{
            .n_tokens = n_tokens,
            .token = batch.token + off,
            .embd = null,
            .pos = batch.pos + off,
            .n_seq_id = batch.n_seq_id + off,
            .seq_id = batch.seq_id + off,
            .logits = batch.logits + off,
        };

        const ret = c.llama_decode(ctx, batch_view);
        if (ret != 0) {
            log.err("failed to decode the batch, n_batch = %d, ret = %d\n", .{ n_batch, ret });
            return false;
        }

        if (synchronize) c.llama_synchronize(ctx);
    }
    return true;
}

/// Ports `llama_batched_bench` (batched-bench.cpp:21 @c1d0e7a00).
fn run(a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    const func = "llama_batched_bench";
    var params = try tc.params.Params.init(a);

    tc.common.init();

    if (!try arg.parse(argv, &params, .bench, printUsage)) return 1;

    const is_pp_shared: c_int = @intFromBool(params.is_pp_shared);
    const is_tg_separate: c_int = @intFromBool(params.is_tg_separate);

    const n_pp = params.n_pp.items;
    const n_tg = params.n_tg.items;
    const n_pl = params.n_pl.items;

    // init LLM

    c.llama_backend_init();
    c.llama_numa_init(params.numa);

    // initialize the model

    const model_params = tc.common.modelParamsToLlama(&params);

    const model_path = try a.dupeSentinel(u8, params.model.path, 0);
    const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse {
        io.eprintf("%s: error: unable to load model\n", .{func});
        return 1;
    };

    var ctx_params = tc.common.contextParamsToLlama(&params);

    // ensure enough sequences are available
    ctx_params.n_seq_max = if (n_pl.len == 0) 1 else @intCast(std.mem.max(i32, n_pl));

    const ctx = c.llama_init_from_model(model, ctx_params) orelse {
        io.eprintf("%s: error: failed to create the llama_context\n", .{func});
        c.llama_model_free(model);
        return 1;
    };

    const vocab = c.llama_model_get_vocab(model);
    const n_vocab = c.llama_vocab_n_tokens(vocab);

    const mem = c.llama_get_memory(ctx);

    const n_kv_max: i32 = @intCast(c.llama_n_ctx(ctx));

    var batch = c.llama_batch_init(n_kv_max, 0, 1);
    const n_batch: i32 = @intCast(ctx_params.n_batch);

    // warm up
    {
        var i: c_int = 0;
        while (i < 16) : (i += 1) tc.common.batchAdd(&batch, @rem(rand(), n_vocab), i, &.{0}, false);

        if (!decodeHelper(ctx, &batch, n_batch, true)) {
            log.err("%s: llama_decode() failed\n", .{func});
            c.llama_free(ctx);
            c.llama_model_free(model);
            return 1;
        }
    }

    if (!params.batched_bench_output_jsonl) {
        log.out("\n", .{});
        log.out("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, is_pp_shared = %d, is_tg_separate = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", .{
            func, n_kv_max, params.n_batch, params.n_ubatch, @as(c_int, @intCast(params.flash_attn_type)), is_pp_shared, is_tg_separate, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
        });
        log.out("\n", .{});
        log.out("|%6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s |\n", .{ "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s" });
        log.out("|%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", .{ "------", "------", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------" });
    }

    for (n_pp) |pp| {
        for (n_tg) |tg| {
            for (n_pl) |pl| {
                const n_ctx_req: i32 = if (is_pp_shared != 0) (if (params.kv_unified) pp else pl * pp) + pl * tg else pl * (pp + tg);

                if (n_ctx_req > n_kv_max) continue;

                tc.common.batchClear(&batch);

                var j: i32 = 0;
                while (j < (if (is_pp_shared != 0) 1 else pl)) : (j += 1) {
                    var i: i32 = 0;
                    while (i < pp) : (i += 1) tc.common.batchAdd(&batch, @rem(rand(), n_vocab), i, &.{j}, i == pp - 1);
                }

                c.llama_memory_clear(mem, false);

                const t_pp_start = c.ggml_time_us();

                if (!decodeHelper(ctx, &batch, n_batch, false)) {
                    log.err("%s: llama_decode() failed\n", .{func});
                    c.llama_free(ctx);
                    c.llama_model_free(model);
                    return 1;
                }

                c.llama_synchronize(ctx);

                const t_pp_end = c.ggml_time_us();

                if (is_pp_shared != 0) {
                    var i: i32 = 1;
                    while (i < pl) : (i += 1) c.llama_memory_seq_cp(mem, 0, i, -1, -1);

                    if (!params.kv_unified) {
                        // run one dummy token to apply the memory copy
                        tc.common.batchClear(&batch);
                        tc.common.batchAdd(&batch, @rem(rand(), n_vocab), pp + 0, &.{0}, true);
                        if (!decodeHelper(ctx, &batch, n_batch, true)) {
                            log.err("%s: llama_decode() failed\n", .{func});
                            c.llama_free(ctx);
                            c.llama_model_free(model);
                            return 1;
                        }
                        _ = c.llama_memory_seq_rm(mem, 0, pp, -1);
                    }
                }

                const t_tg_start = c.ggml_time_us();

                if (is_tg_separate != 0) {
                    // decode pattern:
                    // 0 0 0 ... 1 1 1 ... 2 2 2 ... 3 3 3 ...
                    var jj: i32 = 0;
                    while (jj < pl) : (jj += 1) {
                        var i: i32 = 0;
                        while (i < tg) : (i += 1) {
                            tc.common.batchClear(&batch);
                            tc.common.batchAdd(&batch, @rem(rand(), n_vocab), pp + i, &.{jj}, true);
                            if (!decodeHelper(ctx, &batch, n_batch, true)) {
                                log.err("%s: llama_decode() failed\n", .{func});
                                c.llama_free(ctx);
                                c.llama_model_free(model);
                                return 1;
                            }
                        }
                    }
                } else {
                    // decode pattern:
                    // 0123 0123 0123 ...
                    var i: i32 = 0;
                    while (i < tg) : (i += 1) {
                        tc.common.batchClear(&batch);
                        var jj: i32 = 0;
                        while (jj < pl) : (jj += 1) tc.common.batchAdd(&batch, @rem(rand(), n_vocab), pp + i, &.{jj}, true);
                        if (!decodeHelper(ctx, &batch, n_batch, true)) {
                            log.err("%s: llama_decode() failed\n", .{func});
                            c.llama_free(ctx);
                            c.llama_model_free(model);
                            return 1;
                        }
                    }
                }

                const t_tg_end = c.ggml_time_us();

                const n_kv = n_ctx_req;

                const t_pp: f32 = @as(f32, @floatFromInt(t_pp_end - t_pp_start)) / 1000000.0;
                const t_tg: f32 = @as(f32, @floatFromInt(t_tg_end - t_tg_start)) / 1000000.0;
                const t = t_pp + t_tg;

                const speed_pp: f32 = if (is_pp_shared != 0) @as(f32, @floatFromInt(pp)) / t_pp else @as(f32, @floatFromInt(pl * pp)) / t_pp;
                const speed_tg: f32 = @as(f32, @floatFromInt(pl * tg)) / t_tg;
                const speed: f32 = @as(f32, @floatFromInt((if (is_pp_shared != 0) pp else pl * pp) + pl * tg)) / t;

                if (params.batched_bench_output_jsonl) {
                    log.out("{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"is_pp_shared\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, \"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, \"t\": %f, \"speed\": %f}\n", .{
                        n_kv_max,  params.n_batch, params.n_ubatch, @as(c_int, @intCast(params.flash_attn_type)), @as(c_int, @intFromBool(params.is_pp_shared)), params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                        pp,        tg,             pl,              n_kv,
                        @as(f64, t_pp), @as(f64, speed_pp), @as(f64, t_tg), @as(f64, speed_tg),
                        @as(f64, t),    @as(f64, speed),
                    });
                } else {
                    log.out("|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n", .{
                        pp, tg, pl, n_kv, @as(f64, t_pp), @as(f64, speed_pp), @as(f64, t_tg), @as(f64, speed_tg), @as(f64, t), @as(f64, speed),
                    });
                }
            }
        }
    }

    log.out("\n", .{});
    c.llama_perf_context_print(ctx);

    c.llama_batch_free(batch);

    c.llama_free(ctx);
    c.llama_model_free(model);

    c.llama_backend_free();

    return 0;
}

/// Ports `main` (tools/batched-bench/main.cpp:3 @c1d0e7a00).
pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    return run(arena, argv);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
