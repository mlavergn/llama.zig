//! The fill-in-the-middle sampler.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-sampler.cpp` (v0.4.0, `5266f24da`).
//! Each declaration below names the C++ it replaces and the line it began at.
//!
//! # The vocabulary, through its C API
//!
//! The C++ calls `llama_vocab` member functions. Each one used here has a
//! public C wrapper that forwards unchanged -- `llama_token_to_piece`,
//! `llama_vocab_is_eog`, `llama_vocab_eot`, `llama_vocab_eos` -- so the port
//! calls those, and `llama-vocab.cpp` stays C++ until libllama's swap.
//!
//! # One inherited termination
//!
//! When a token's text is longer than the 512-byte buffer, the C++ calls
//! `buf.resize(len)` with the **negative** length `token_to_piece` returned.
//! That converts to a huge `size_t`, `std::vector` throws `length_error`,
//! and nothing catches it. The port terminates at the same point with the
//! same exception text rather than "fixing" a path no shipped vocabulary
//! reaches -- see "Where the C's own answer is unspecified" in `CLAUDE.md`.

const std = @import("std");
const impl = @import("../impl.zig");
const core = @import("core.zig");

const c = impl.c;
const Sampler = core.Sampler;
const SamplerI = core.SamplerI;
const TokenDataArray = core.TokenDataArray;
const src = core.src;

/// Ports `llama_sampler_infill` (llama-sampler.cpp:4070 @5266f24da).
const Infill = struct {
    vocab: ?*const c.struct_llama_vocab,

    buf0: [512]u8,
    buf1: [512]u8,
};

/// Ports `llama_sampler_infill_name` (llama-sampler.cpp:4077 @5266f24da).
fn infillName(smpl: [*c]const Sampler) callconv(.c) [*c]const u8 {
    _ = smpl;
    return "infill";
}

/// `vocab->token_to_piece(id, buf, size, 0, false)`, terminating as the C++
/// does when the piece does not fit -- see the file header.
fn pieceInto(vocab: ?*const c.struct_llama_vocab, id: c.llama_token, buf: []u8) usize {
    const len = c.llama_token_to_piece(vocab, id, buf.ptr, @intCast(buf.len), 0, false);
    if (len < 0) {
        c.ggml_abort(src, 4152, "terminating due to uncaught exception of type std::length_error: %s", "vector");
    }
    return @intCast(len);
}

/// Ports `llama_sampler_infill_apply` (llama-sampler.cpp:4081 @5266f24da).
///
/// Picks end-of-generation when its mass is high enough; otherwise merges
/// each candidate into any it is a text prefix of, then drops the weak ones
/// in two rounds, falling back to EOT (or EOS) when nothing non-EOG is left.
fn infillApply(smpl: [*c]Sampler, cur_p: [*c]TokenDataArray) callconv(.c) void {
    const ctx = core.ctxOf(Infill, smpl);

    core.softmaxImpl(cur_p, true);

    var p_txt_sum: f32 = 0.0;
    var p_eog_sum: f32 = 0.0;

    for (core.items(cur_p)) |d| {
        if (c.llama_vocab_is_eog(ctx.vocab, d.id)) {
            p_eog_sum += d.p;
        } else {
            p_txt_sum += d.p;
        }
    }

    if (3 * p_eog_sum * @as(f32, @floatFromInt(cur_p.*.size)) > p_txt_sum) {
        // Keep just the EOG tokens.
        const size_org = cur_p.*.size;

        cur_p.*.size = 0;

        var p_sum: f32 = 0.0;

        for (0..size_org) |i| {
            if (c.llama_vocab_is_eog(ctx.vocab, cur_p.*.data[i].id)) {
                p_sum += cur_p.*.data[i].p;

                cur_p.*.data[cur_p.*.size] = cur_p.*.data[i];
                cur_p.*.size += 1;
            }
        }

        for (core.items(cur_p)) |*d| d.p /= p_sum;

        return;
    }

    // Combine tokens with a common prefix.
    for (0..cur_p.*.size) |i_0| {
        for (0..cur_p.*.size) |i_1| {
            if (cur_p.*.data[i_0].logit == -std.math.inf(f32)) break;

            if (i_0 == i_1 or cur_p.*.data[i_1].logit == -std.math.inf(f32)) continue;

            const len0 = pieceInto(ctx.vocab, cur_p.*.data[i_0].id, &ctx.buf0);
            const len1 = pieceInto(ctx.vocab, cur_p.*.data[i_1].id, &ctx.buf1);

            // Token i0 is a prefix of token i1.
            if (len0 > 0 and len0 <= len1 and std.mem.eql(u8, ctx.buf0[0..len0], ctx.buf1[0..len0])) {
                var dst = i_0;
                var srci = i_1;

                // Merge into the more probable of the two.
                if (cur_p.*.data[i_1].p > cur_p.*.data[i_0].p) {
                    std.mem.swap(usize, &dst, &srci);
                }

                cur_p.*.data[dst].p += cur_p.*.data[srci].p;
                cur_p.*.data[srci].logit = -std.math.inf(f32);
                cur_p.*.data[srci].p = 0.0;
            }
        }
    }

    var n_non_eog: usize = 0;

    var size_org = cur_p.*.size;

    var p_sum: f32 = 0.0;
    var thold: f32 = 0.2;

    cur_p.*.size = 0;

    for (0..size_org) |i| {
        const is_eog = c.llama_vocab_is_eog(ctx.vocab, cur_p.*.data[i].id);

        if (cur_p.*.data[i].p < thold and !is_eog) continue;

        if (!is_eog) n_non_eog += 1;

        p_sum += cur_p.*.data[i].p;

        cur_p.*.data[cur_p.*.size] = cur_p.*.data[i];
        cur_p.*.size += 1;
    }

    if (n_non_eog == 0) {
        cur_p.*.size = 1;
        cur_p.*.data[0].id = c.llama_vocab_eot(ctx.vocab);
        if (cur_p.*.data[0].id == c.LLAMA_TOKEN_NULL) {
            cur_p.*.data[0].id = c.llama_vocab_eos(ctx.vocab);
        }
        cur_p.*.data[0].logit = 1.0;

        impl.assert(cur_p.*.data[0].id != c.LLAMA_TOKEN_NULL, src, 4222, "cur_p->data[0].id != LLAMA_TOKEN_NULL");

        return;
    }

    for (core.items(cur_p)) |*d| d.p /= p_sum;

    size_org = cur_p.*.size;
    p_sum = 0.0;
    thold = @floatCast(1.0 / @as(f64, @floatFromInt(n_non_eog + 1)));

    cur_p.*.size = 0;

    for (0..size_org) |i| {
        const is_eog = c.llama_vocab_is_eog(ctx.vocab, cur_p.*.data[i].id);

        if (cur_p.*.data[i].p < thold and !is_eog) continue;

        p_sum += cur_p.*.data[i].p;

        cur_p.*.data[cur_p.*.size] = cur_p.*.data[i];
        cur_p.*.size += 1;
    }

    for (core.items(cur_p)) |*d| d.p /= p_sum;
}

/// Ports `llama_sampler_infill_clone` (llama-sampler.cpp:4264 @5266f24da).
fn infillClone(smpl: [*c]const Sampler) callconv(.c) [*c]Sampler {
    return llama_sampler_init_infill(core.ctxOf(Infill, smpl).vocab);
}

/// Ports `llama_sampler_infill_free` (llama-sampler.cpp:4269 @5266f24da).
fn infillFree(smpl: [*c]Sampler) callconv(.c) void {
    core.destroy(Infill, core.ctxOf(Infill, smpl));
}

/// Ports `llama_sampler_infill_i` (llama-sampler.cpp:4273 @5266f24da).
var infill_iface: SamplerI = .{
    .name = infillName,
    .accept = null,
    .apply = infillApply,
    .reset = null,
    .clone = infillClone,
    .free = infillFree,
};

/// Ports `llama_sampler_init_infill` (llama-sampler.cpp:4288 @5266f24da).
///
/// Parameters:
/// - `vocab`: the vocabulary; borrowed for the sampler's lifetime.
///
/// Return: the sampler.
pub export fn llama_sampler_init_infill(vocab: ?*const c.struct_llama_vocab) callconv(.c) [*c]Sampler {
    return core.llama_sampler_init(&infill_iface, core.create(Infill, .{
        .vocab = vocab,
        .buf0 = std.mem.zeroes([512]u8),
        .buf1 = std.mem.zeroes([512]u8),
    }));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
