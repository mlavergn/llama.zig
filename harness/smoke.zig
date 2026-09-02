//! End-to-end check that the reference build actually runs.
//!
//! Loads a GGUF model through the C ABI of the vendored libllama, offloads it
//! to Metal, and greedily decodes a few tokens. A green `zig build reference`
//! only proves the tree compiles and links; this proves the result infers.
//!
//! It is also a rehearsal for the port. Every file ported in Stages 3 and 4
//! keeps its C ABI so the C++ above it links unchanged, and this driver
//! exercises that same boundary from the Zig side.
//!
//! Usage: zig build smoke -- <model.gguf> [prompt] [n_predict]

const std = @import("std");

const c = @cImport({
    @cInclude("llama.h");
});

/// How many tokens to generate when the caller does not say.
const default_predict: usize = 24;

/// The prompt used when the caller does not supply one.
const default_prompt: []const u8 = "The capital of France is";

/// Largest prompt this accepts, in tokens. Small on purpose: this is a smoke
/// test, not a CLI.
const max_prompt_tokens = 512;

/// Loads a model, decodes greedily, and writes the completion to stdout.
///
/// Parameters:
/// - `init`: process capabilities supplied by the runtime (allocator, IO, args).
///
/// Return: nothing on success. Returns `error.MissingModelPath` when no model
/// is given, and surfaces libllama failures as distinct errors rather than a
/// single opaque one, so a failure names the stage it happened in.
pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) return error.MissingModelPath;

    const model_path = args[1];
    const prompt: []const u8 = if (args.len > 2) args[2] else default_prompt;
    const n_predict: usize = if (args.len > 3)
        try std.fmt.parseInt(usize, args[3], 10)
    else
        default_predict;

    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    const out = &stdout.interface;

    c.llama_backend_init();
    defer c.llama_backend_free();

    var model_params = c.llama_model_default_params();
    // The whole point of this check is that the Metal path works, so the model
    // goes entirely to the GPU rather than falling back to CPU on a bad build.
    model_params.n_gpu_layers = 99;

    const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse
        return error.ModelLoadFailed;
    defer c.llama_model_free(model);

    const vocab = c.llama_model_get_vocab(model) orelse return error.NoVocab;

    var tokens: [max_prompt_tokens]c.llama_token = undefined;
    const n_prompt = c.llama_tokenize(
        vocab,
        prompt.ptr,
        @intCast(prompt.len),
        &tokens,
        max_prompt_tokens,
        true,
        true,
    );
    if (n_prompt < 0) return error.TokenizeFailed;

    var ctx_params = c.llama_context_default_params();
    ctx_params.n_ctx = max_prompt_tokens;
    ctx_params.n_batch = max_prompt_tokens;

    const ctx = c.llama_init_from_model(model, ctx_params) orelse return error.ContextFailed;
    defer c.llama_free(ctx);

    const sampler = c.llama_sampler_chain_init(c.llama_sampler_chain_default_params()) orelse
        return error.SamplerFailed;
    defer c.llama_sampler_free(sampler);
    // Greedy, so the output is a function of the model and prompt alone. A
    // sampler with any randomness would make this check unable to fail loudly.
    c.llama_sampler_chain_add(sampler, c.llama_sampler_init_greedy());

    try out.print("{s}", .{prompt});
    try out.flush();

    // `llama_batch_get_one` borrows the token buffer rather than copying it,
    // and the batch built at the end of one iteration is not read until the
    // decode at the top of the next. So the sampled token lives out here,
    // outside the loop, where its lifetime covers both.
    var id: c.llama_token = undefined;

    var batch = c.llama_batch_get_one(&tokens, n_prompt);
    var generated: usize = 0;
    while (generated < n_predict) : (generated += 1) {
        if (c.llama_decode(ctx, batch) != 0) return error.DecodeFailed;

        id = c.llama_sampler_sample(sampler, ctx, -1);
        if (c.llama_vocab_is_eog(vocab, id)) break;

        var piece: [256]u8 = undefined;
        const n = c.llama_token_to_piece(vocab, id, &piece, piece.len, 0, true);
        if (n > 0) {
            try out.print("{s}", .{piece[0..@intCast(n)]});
            try out.flush();
        }

        batch = c.llama_batch_get_one(&id, 1);
    }

    try out.print("\n", .{});
    try out.flush();
}
