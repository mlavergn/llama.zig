//! The per-model encoder graphs (`llama.cpp/tools/mtmd/models/`), and
//! `clip_get_graph_builder` (`llama.cpp/tools/mtmd/clip.cpp`, v0.4.0,
//! `5266f24da`), which picks one by projector type.
//!
//! Each file here ports the `models/*.cpp` of the same name (hyphens become
//! underscores) and exposes `build(g, params)`: the class's `build()`, run
//! on a `graph.Graph` set up by `buildGraph`. A model that overrides
//! `build_mm` sets `g.mm_hook` first; `support_batch()` is the table below,
//! since the C++ asks it of a builder that never builds.

const std = @import("std");
const impl = @import("../impl.zig");
const clip = @import("../clip.zig");
const graph = @import("../graph.zig");
const model_mod = @import("../model.zig");
const names = @import("../names.zig");
const c = impl.c;

pub const cogvlm = @import("cogvlm.zig");
pub const conformer = @import("conformer.zig");
pub const deepseek4v = @import("deepseek4v.zig");
pub const deepseekocr = @import("deepseekocr.zig");
pub const deepseekocr2 = @import("deepseekocr2.zig");
pub const dots3note = @import("dots3note.zig");
pub const dotsocr = @import("dotsocr.zig");
pub const exaone4_5 = @import("exaone4_5.zig");
pub const gemma4a = @import("gemma4a.zig");
pub const gemma4ua = @import("gemma4ua.zig");
pub const gemma4uv = @import("gemma4uv.zig");
pub const gemma4v = @import("gemma4v.zig");
pub const glm4v = @import("glm4v.zig");
pub const granite_speech = @import("granite_speech.zig");
pub const granite4_vision = @import("granite4_vision.zig");
pub const hunyuanvl = @import("hunyuanvl.zig");
pub const internvl = @import("internvl.zig");
pub const kimik25 = @import("kimik25.zig");
pub const kimivl = @import("kimivl.zig");
pub const llama4 = @import("llama4.zig");
pub const llava = @import("llava.zig");
pub const mimo_audio = @import("mimo_audio.zig");
pub const mimovl = @import("mimovl.zig");
pub const minicpmv = @import("minicpmv.zig");
pub const minimax_m3 = @import("minimax_m3.zig");
pub const mobilenetv5 = @import("mobilenetv5.zig");
pub const muse_glimmer = @import("muse_glimmer.zig");
pub const nemotron_v2_vl = @import("nemotron_v2_vl.zig");
pub const paddleocr = @import("paddleocr.zig");
pub const parakeet = @import("parakeet.zig");
pub const pixtral = @import("pixtral.zig");
pub const pockettts_gen = @import("pockettts_gen.zig");
pub const pockettts_seanet = @import("pockettts_seanet.zig");
pub const pockettts_spkenc = @import("pockettts_spkenc.zig");
pub const qwen2vl = @import("qwen2vl.zig");
pub const qwen3a = @import("qwen3a.zig");
pub const qwen3tts_gen = @import("qwen3tts_gen.zig");
pub const qwen3tts_spkenc = @import("qwen3tts_spkenc.zig");
pub const qwen3vl = @import("qwen3vl.zig");
pub const siglip = @import("siglip.zig");
pub const step3vl = @import("step3vl.zig");
pub const whisper_enc = @import("whisper_enc.zig");
pub const yasa2 = @import("yasa2.zig");
pub const youtuvl = @import("youtuvl.zig");

/// What `clip_get_graph_builder(...)->build()` leaves: the graph, and
/// whether the builder supports batching. `deinit` frees the builder's
/// context, as the C++'s temporary does at the end of the expression; the
/// graph itself lives in `buf_compute_meta` and outlives it.
pub const Built = struct {
    gf: ?*c.ggml_cgraph,
    support_batch: bool,
    ctx0: ?*c.ggml_context,

    pub fn deinit(self: *Built) void {
        if (self.ctx0 != null) c.ggml_free(self.ctx0);
        self.ctx0 = null;
    }
};

/// The `support_batch()` override of each builder (models.h @5266f24da),
/// or null for a projector `clip_get_graph_builder` has no builder for.
pub fn supportBatch(t: names.ProjectorType) ?bool {
    return switch (t) {
        .gemma3, .idefics3, .lfm2, .janus_pro, .phi4 => false,
        .gemma3nv => false,
        .gemma4v => true,
        .gemma4uv => false,
        .pixtral, .lightonocr => false,
        .dots_ocr, .dots3note_v => false,
        .dots3note_a => false,
        .qwen2vl, .qwen25vl => false,
        .qwen3vl => false,
        .exaone4_5 => false,
        .mimovl => false,
        .minimax_m3 => false,
        .muse_glimmer => false,
        .step3vl => false,
        .minicpmv => false,
        .minicpmv4_6 => false,
        .internvl => true,
        .nemotron_v2_vl => false,
        .llama4 => false,
        .ultravox, .voxtral, .qwen2a, .glma, .meralion, .music_flamingo => false,
        .kimivl => false,
        .paddleocr => false,
        .kimik25 => false,
        .deepseek4v => false,
        .cogvlm => false,
        .hunyuanvl => false,
        .mlp, .mlp_norm, .ldp, .ldpv2, .glm_edge => false,
        .deepseekocr => true,
        .deepseekocr2 => true,
        .lfm2a => false,
        .gemma4a => false,
        .gemma4ua => false,
        .granite_speech => false,
        .glm4v => false,
        .qwen3a => false,
        .mimo_audio => false,
        .qwen3tts_spkenc => false,
        .pockettts_spkenc => false,
        .pockettts_gen => false,
        .qwen3tts_gen => false,
        .youtuvl => false,
        .yasa2 => false,
        .parakeet => false,
        .granite4_vision => false,
        else => null,
    };
}

/// Ports `clip_get_graph_builder` (mtmd/clip.cpp:930 @5266f24da) and the
/// builder's `build()`.
pub fn buildGraph(ctx: *clip.Ctx, imgs: *const impl.ImageF32Batch, params: ?*const clip.EncodeParams) clip.Error!Built {
    const img = &imgs.entries.items[0];
    var g = graph.Graph.init(ctx.graphSource(), img);
    g.img_batch = imgs;
    // TODO [QWEN_VIDEO]: improve this in the future
    g.n_batch = @intCast(imgs.entries.items.len);
    const gf: ?*c.ggml_cgraph = switch (ctx.projType()) {
        .gemma3, .idefics3, .lfm2, .janus_pro, .phi4 => try siglip.build(&g, params),
        .gemma3nv => try mobilenetv5.build(&g, params),
        .gemma4v => try gemma4v.build(&g, params),
        .gemma4uv => try gemma4uv.build(&g, params),
        .pixtral, .lightonocr => try pixtral.build(&g, params),
        .dots_ocr, .dots3note_v => try dotsocr.build(&g, params),
        .dots3note_a => try dots3note.build(&g, params),
        .qwen2vl, .qwen25vl => try qwen2vl.build(&g, params),
        .qwen3vl => try qwen3vl.build(&g, params),
        .exaone4_5 => try exaone4_5.build(&g, params),
        .mimovl => try mimovl.build(&g, params),
        .minimax_m3 => try minimax_m3.build(&g, params),
        .muse_glimmer => try muse_glimmer.build(&g, params),
        .step3vl => try step3vl.build(&g, params),
        .minicpmv => try minicpmv.build(&g, params),
        .minicpmv4_6 => try minicpmv.build4_6(&g, params),
        .internvl => try internvl.build(&g, params),
        .nemotron_v2_vl => try nemotron_v2_vl.build(&g, params),
        .llama4 => try llama4.build(&g, params),
        .ultravox, .voxtral, .qwen2a, .glma, .meralion, .music_flamingo => try whisper_enc.build(&g, params),
        .kimivl => try kimivl.build(&g, params),
        .paddleocr => try paddleocr.build(&g, params),
        .kimik25 => try kimik25.build(&g, params),
        .deepseek4v => try deepseek4v.build(&g, params),
        .cogvlm => try cogvlm.build(&g, params),
        .hunyuanvl => try hunyuanvl.build(&g, params),
        .mlp, .mlp_norm, .ldp, .ldpv2, .glm_edge => try llava.build(&g, params),
        .deepseekocr => try deepseekocr.build(&g, params),
        .deepseekocr2 => try deepseekocr2.build(&g, params),
        .lfm2a => try conformer.build(&g, params),
        .gemma4a => try gemma4a.build(&g, params),
        .gemma4ua => try gemma4ua.build(&g, params),
        .granite_speech => try granite_speech.build(&g, params),
        .glm4v => try glm4v.build(&g, params),
        .qwen3a => try qwen3a.build(&g, params),
        .mimo_audio => try mimo_audio.build(&g, params),
        .qwen3tts_spkenc => try qwen3tts_spkenc.build(&g, params),
        .pockettts_spkenc => try pockettts_spkenc.build(&g, params),
        .pockettts_gen => try pockettts_gen.build(&g, params),
        .qwen3tts_gen => try qwen3tts_gen.build(&g, params),
        .youtuvl => try youtuvl.build(&g, params),
        .yasa2 => try yasa2.build(&g, params),
        .parakeet => try parakeet.build(&g, params),
        .granite4_vision => try granite4_vision.build(&g, params),
        else => impl.abort("clip.cpp", 1143, "missing cgraph builder"),
    };
    return .{ .gf = gf, .support_batch = supportBatch(ctx.projType()).?, .ctx0 = g.ctx0 };
}

/// Ports `c2w_state_slot` (models/qwen3tts-gen.cpp @5266f24da): a persisted
/// state slot of a gen-audio decoder.
pub const StateSlot = struct {
    name: [:0]const u8,
    ne0: i64,
    ne1: i64,
};

/// Ports `list_gen_state_slots` (mtmd/clip.cpp:4408 @5266f24da). The caller
/// frees the slice with `freeSlots`.
pub fn genStateSlots(hparams: *const model_mod.Hparams, model: *const model_mod.Model) clip.Error![]StateSlot {
    return switch (model.proj_type) {
        .qwen3tts_gen => qwen3tts_gen.listStateSlots(hparams, model),
        .pockettts_gen => pockettts_gen.listStateSlots(hparams, model),
        else => &.{},
    };
}

pub fn freeSlots(slots: []StateSlot) void {
    if (slots.len == 0) return;
    for (slots) |s| impl.gpa.free(s.name);
    impl.gpa.free(slots);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
