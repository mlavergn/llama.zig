//! `llama-tts`: speak a prompt through a backbone model and an mmproj that
//! generates audio.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/tts/tts.cpp` at v0.4.1 (`b29c606e2`). Each
//! declaration names the C++ it replaces and the line it began at.
//!
//! The C++ holds the mtmd context, the speaker bitmap and the generation
//! helper in RAII wrappers declared in that order after `llama_init`; the
//! `defer`s below free them in the reverse order the destructors run.

const std = @import("std");
const tc = @import("tools_common");
const c = tc.c;
const log = tc.log;
const arg = tc.arg;

comptime {
    // libllama and libmtmd, compiled in (see `build/tools.zig`).
    _ = @import("llama");
    _ = @import("mtmd");
}

extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*c.FILE;
extern fn fwrite(p: ?*const anyopaque, size: usize, n: usize, f: ?*c.FILE) usize;
extern fn fclose(f: ?*c.FILE) c_int;

/// Ports `tts_timings` (tts.cpp:19 @b29c606e2): a progress line every two
/// seconds.
const Timings = struct {
    t_start_us: i64,
    t_last_us: i64,

    fn init() Timings {
        const t = c.ggml_time_us();
        return .{ .t_start_us = t, .t_last_us = t };
    }

    /// `report` (tts.cpp:23 @b29c606e2), a member of `tts_timings`.
    fn report(self: *Timings, n_frames: c_int) void {
        const t_now_us = c.ggml_time_us();
        if (t_now_us - self.t_last_us < 2000000) return;
        self.t_last_us = t_now_us;
        const t_elapsed_s = @as(f64, @floatFromInt(t_now_us - self.t_start_us)) / 1e6;
        const fps = if (t_elapsed_s > 0) @as(f64, @floatFromInt(n_frames)) / t_elapsed_s else 0.0;
        log.inf("frames generated: %d, speed: %.2f frames/s\n", .{ n_frames, fps });
    }
};

/// Ports `print_usage` (tts.cpp:35 @b29c606e2).
fn printUsage(argv: []const [:0]const u8) void {
    log.out("\nexample usage:\n", .{});
    log.out("\n    %s -m backbone.gguf -mm mmproj.gguf -p \"text to speak\" -o output.wav", .{argv[0].ptr});
    log.out("\n    %s -hf user/model -p \"text to speak\" -o output.wav\n", .{argv[0].ptr});
    log.out("\nnote: --tts-lang and --tts-speaker-file may not be supported in all models", .{});
    log.out("\n      use -n to limit the output length", .{});
    log.out("\n      see tts/README.md for per-model usage notes", .{});
    log.out("\n\n", .{});
}

fn run(init: std.process.Init, a: std.mem.Allocator, argv: []const [:0]const u8) !u8 {
    var params = try tc.params.Params.init(a);

    tc.common.init();

    if (!try arg.parse(argv, &params, .tts, printUsage)) return 1;

    c.mtmd_helper_log_set(tc.common.defaultLogCallback, null);

    if (params.prompt.len == 0) {
        log.err("no prompt provided, use -p \"text\"\n", .{});
        return 1;
    }
    if (params.mmproj.path.len == 0) {
        log.err("no mmproj provided, use --mmproj\n", .{});
        return 1;
    }

    // important: keep this file as generic as possible
    //            model-specific logic should be in mtmd-helper-gen or mtmd API

    // always enable embd, so that we can pass hidden states to the audio generation helper
    params.embedding = true;

    c.llama_backend_init();
    c.llama_numa_init(params.numa);

    //
    // load backbone model and mmproj
    //

    var llama_init = try tc.init.initFromParams(init.gpa, &params, false);
    defer llama_init.deinit();
    const model = llama_init.model;
    const lctx = llama_init.context;
    const smpl = llama_init.sampler(0);
    if (model == null or lctx == null) {
        log.err("failed to init model/context\n", .{});
        return 1;
    }

    const mmproj_path = try a.dupeSentinel(u8, params.mmproj.path, 0);
    var mtmd_params = c.mtmd_context_params_default();
    mtmd_params.use_gpu = params.mmproj_use_gpu;
    mtmd_params.device = params.mmproj_device;
    const mctx = c.mtmd_init_from_file(mmproj_path.ptr, model, mtmd_params) orelse {
        log.err("failed to load mmproj %s\n", .{mmproj_path.ptr});
        return 1;
    };
    defer c.mtmd_free(mctx);
    if (c.mtmd_gen_audio_get_info(mctx).type == c.MTMD_GEN_AUDIO_TYPE_NONE) {
        log.err("mmproj does not support audio generation\n", .{});
        return 1;
    }

    //
    // stage 0: process speaker reference file, if any
    //

    var speaker_bitmap: ?*c.mtmd_bitmap = null;
    defer if (speaker_bitmap) |bm| c.mtmd_bitmap_free(bm);
    if (params.tts_speaker_file.len != 0) {
        const speaker_file = try a.dupeSentinel(u8, params.tts_speaker_file, 0);
        const wrapper = c.mtmd_helper_bitmap_init_from_file(mctx, speaker_file.ptr, false, c.mtmd_helper_init_opt_default());
        if (wrapper.bitmap == null) {
            log.err("failed to load speaker file %s\n", .{speaker_file.ptr});
            return 1;
        }
        speaker_bitmap = wrapper.bitmap;
    }

    const gen = c.mtmd_helper_gen_audio_init(lctx, mctx);
    defer c.mtmd_helper_gen_audio_free(gen);
    const prompt = try a.dupeSentinel(u8, params.prompt, 0);
    const lang = try a.dupeSentinel(u8, params.tts_lang, 0);
    var inp = std.mem.zeroes(c.mtmd_helper_gen_audio_inp);
    inp.seq_id = 0;
    inp.prompt = prompt.ptr;
    inp.prompt_len = prompt.len;
    inp.speaker_ref = speaker_bitmap;
    inp.lang = lang.ptr;
    inp.top_k = params.sampling.top_k;
    inp.top_p = params.sampling.top_p;
    inp.seed = params.sampling.seed;
    inp.out_type = c.MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;

    //
    // stage 1: process prompt via backbone model, generate semantic representation
    //

    if (c.mtmd_helper_gen_audio_set_input(gen, &inp) != 0) {
        log.err("set_input failed\n", .{});
        return 1;
    }

    const t_prompt_start_us = c.ggml_time_us();

    while (true) {
        const ret = c.mtmd_helper_gen_audio_step_prompt(gen, params.n_batch);
        if (ret < 0) {
            log.err("prompt processing failed\n", .{});
            return 1;
        }
        if (ret == 0) break;
    }

    // note: some pipelines ignore this token and use the hidden state instead
    const Sample = struct {
        fn semanticCode(s: ?*tc.sampling.Sampler, ctx: ?*c.llama_context) !c.llama_token {
            const t = try s.?.sample(ctx, -1, false);
            s.?.accept(t, true);
            return t;
        }
    };

    const max_new: c_int = if (params.n_predict > 0) params.n_predict else 512;
    var n_frames: c_int = 0;
    var sampled = try Sample.semanticCode(smpl, lctx);
    var h_state: ?[*]const f32 = c.llama_get_embeddings_ith(lctx, -1);

    var timings = Timings.init();
    const t_gen_start_us = c.ggml_time_us();

    var stop = false;
    while (!stop and n_frames < max_new) {
        var h_next: [*c]const f32 = null;

        // stage 2+3: semantic --> acoustic details --> audio waveform
        //            step_gen() runs both stages and returns new h_state for next step
        if (c.mtmd_helper_gen_audio_step_gen(gen, sampled, h_state, &h_next, &stop) != 0) {
            log.err("step_gen failed at frame %d\n", .{n_frames});
            return 1;
        }
        if (h_next == null) break; // stopped without generating a frame

        n_frames += 1;
        h_state = h_next;
        sampled = try Sample.semanticCode(smpl, lctx);
        timings.report(n_frames);
    }
    const t_gen_s = @as(f64, @floatFromInt(c.ggml_time_us() - t_gen_start_us)) / 1e6;

    var sample_rate: i32 = 0;
    var data: [*c]const u8 = null;
    var data_len: usize = 0;
    var n_samples: i64 = 0;
    const t_wav_start_us = c.ggml_time_us();
    if (c.mtmd_helper_gen_audio_get_output(gen, &sample_rate, &data, &data_len, &n_samples) != 0) {
        log.err("get_output failed\n", .{});
        return 1;
    }
    const t_wav_s = @as(f64, @floatFromInt(c.ggml_time_us() - t_wav_start_us)) / 1e6;

    log.inf("generated %d frames, %zu bytes of WAV audio (%d Hz)\n", .{ n_frames, data_len, sample_rate });

    const t_prompt_s = @as(f64, @floatFromInt(t_gen_start_us - t_prompt_start_us)) / 1e6;
    const t_total_s = t_prompt_s + t_gen_s + t_wav_s;
    const audio_s = if (sample_rate > 0) @as(f64, @floatFromInt(n_samples)) / @as(f64, @floatFromInt(sample_rate)) else 0.0;
    log.inf("timings: prompt eval %.2fs + generation %.2fs + vocoder %.2fs = total %.2fs\n", .{ t_prompt_s, t_gen_s, t_wav_s, t_total_s });
    log.inf("         output audio = %.2fs (audio time = %.2fx process time)\n", .{ audio_s, if (t_total_s > 0) audio_s / t_total_s else 0.0 });
    const out_file = try a.dupeSentinel(u8, params.out_file, 0);
    const f = fopen(out_file.ptr, "wb") orelse {
        log.err("failed to open %s\n", .{out_file.ptr});
        return 1;
    };
    _ = fwrite(data, 1, data_len, f);
    _ = fclose(f);
    log.inf("wrote %s\n", .{out_file.ptr});

    c.llama_backend_free();
    return 0;
}

/// Ports `main` (tts.cpp:45 @b29c606e2).
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
