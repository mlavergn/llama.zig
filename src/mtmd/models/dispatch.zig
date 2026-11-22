//! `mtmd_get_graph_builder` (`llama.cpp/tools/mtmd/mtmd-graph.cpp`, v1.0.0,
//! `b1b770d90`), which picks a model's builder by projector type, with
//! `list_gen_state_slots` and `list_enc_state_slots`
//! (`llama.cpp/tools/mtmd/mtmd-subcontext.cpp`), which list the state slots
//! of the builders that keep state between calls. `support_batch()` is the
//! table below, since the C++ asks it of a builder that never builds.

const std = @import("std");
const llama = @import("llama");
const mtmd = @import("../module.zig");

/// What `mtmd_get_graph_builder(...)->build()` leaves: the graph, and
/// whether the builder supports batching. `deinit` frees the builder's
/// context, as the C++'s temporary does at the end of the expression; the
/// graph itself lives in `buf_compute_meta` and outlives it.
pub const Built = struct {
    gf: ?*llama.impl.llama_c.ggml_cgraph,
    support_batch: bool,
    ctx0: ?*llama.impl.llama_c.ggml_context,

    pub fn deinit(self: *Built) void {
        if (self.ctx0 != null) llama.impl.llama_c.ggml_free(self.ctx0);
        self.ctx0 = null;
    }
};

/// The `support_batch()` override of each builder (models.h @b1b770d90),
/// or null for a projector `mtmd_get_graph_builder` has no builder for.
pub fn supportBatch(t: mtmd.names.ProjectorType) ?bool {
    return switch (t) {
        .gemma3, .idefics3, .cohere2v, .lfm2, .d1omni_v, .janus_pro, .phi4 => false,
        .gemma3nv => false,
        .gemma4v => true,
        .gemma4uv => false,
        .pixtral, .lightonocr => false,
        .dots_ocr, .dots3note_v => false,
        .dots3note_a => false,
        .qwen2vl, .qwen25vl => false,
        .qwen3vl => false,
        .magevl => false,
        .ling3vl => false,
        .exaone4_5 => false,
        .mimovl => false,
        .minimax_m3 => false,
        .muse_glimmer => false,
        .step3vl => false,
        .minicpmv => false,
        .minicpmv4_6, .minicpmv4_7, .minicpm_ocr_1_0 => false,
        .internvl => true,
        .nemotron_v2_vl => false,
        .nemotron_omni => false,
        .llama4 => false,
        .ultravox, .voxtral, .qwen2a, .glma, .meralion, .music_flamingo => false,
        .kimivl => false,
        .paddleocr => false,
        .kimik25, .kimik31 => false,
        .deepseek4v, .deepseek41v => false,
        .cogvlm => false,
        .hunyuanvl => false,
        .mlp, .mlp_norm, .ldp, .ldpv2, .glm_edge => false,
        .deepseekocr => true,
        .deepseekocr2 => true,
        .lfm2a, .d1omni_a => false,
        .gemma4a => false,
        .gemma4ua => false,
        .granite_speech => false,
        .glm4v, .glm5v => false,
        .qwen3a => false,
        .mimo_audio => false,
        .qwen3tts_spkenc => false,
        .pockettts_spkenc => false,
        .chatterbox_spkenc => false,
        .chatterbox => false,
        .pockettts_gen => false,
        .qwen3tts_gen => false,
        .youtuvl => false,
        .yasa2 => false,
        .parakeet => false,
        .nemotron_transcribe => false,
        .granite4_vision => false,
        .lfm3a => false,
        .soprano => false,
        .sensenova_u1 => false,
        .lfm3vl => false,
        .voxtral_rt => false,
        .muse_spark_audio => false,
        .lfm3a_gen => false,
        .drama3_gen => false,
        else => null,
    };
}

/// Ports `mtmd_get_graph_builder` (mtmd-graph.cpp:749 @b1b770d90) and the
/// builder's `build()`.
pub fn buildGraph(ctx: *mtmd.clip.Ctx, imgs: *const mtmd.impl.ImageF32Batch, params: ?*const mtmd.clip.EncodeParams) mtmd.clip.Error!Built {
    const img = &imgs.entries.items[0];
    var g = mtmd.graph.Graph.init(ctx.graphSource(), img);
    g.img_batch = imgs;
    // TODO [QWEN_VIDEO]: improve this in the future
    g.n_batch = @intCast(imgs.entries.items.len);
    const gf: ?*llama.impl.llama_c.ggml_cgraph = switch (ctx.projType()) {
        .gemma3, .idefics3, .cohere2v, .lfm2, .d1omni_v, .janus_pro, .phi4 => try mtmd.models.siglip.build(&g, params),
        .gemma3nv => try mtmd.models.mobilenetv5.build(&g, params),
        .gemma4v => try mtmd.models.gemma4v.build(&g, params),
        .gemma4uv => try mtmd.models.gemma4uv.build(&g, params),
        .pixtral, .lightonocr => try mtmd.models.pixtral.build(&g, params),
        .dots_ocr, .dots3note_v => try mtmd.models.dotsocr.build(&g, params),
        .dots3note_a => try mtmd.models.dots3note.build(&g, params),
        .qwen2vl, .qwen25vl => try mtmd.models.qwen2vl.build(&g, params),
        .qwen3vl => try mtmd.models.qwen3vl.build(&g, params),
        .magevl => try mtmd.models.magevl.build(&g, params),
        .ling3vl => try mtmd.models.ling3vl.build(&g, params),
        .exaone4_5 => try mtmd.models.exaone4_5.build(&g, params),
        .mimovl => try mtmd.models.mimovl.build(&g, params),
        .minimax_m3 => try mtmd.models.minimax_m3.build(&g, params),
        .muse_glimmer => try mtmd.models.muse_glimmer.build(&g, params),
        .step3vl => try mtmd.models.step3vl.build(&g, params),
        .minicpmv => try mtmd.models.minicpmv.build(&g, params),
        .minicpmv4_6, .minicpmv4_7, .minicpm_ocr_1_0 => try mtmd.models.minicpmv.build4_6(&g, params),
        .internvl => try mtmd.models.internvl.build(&g, params),
        .nemotron_v2_vl => try mtmd.models.nemotron_v2_vl.build(&g, params),
        .nemotron_omni => try mtmd.models.nemotron_omni.build(&g, params),
        .llama4 => try mtmd.models.llama4.build(&g, params),
        .ultravox, .voxtral, .qwen2a, .glma, .meralion, .music_flamingo => try mtmd.models.whisper_enc.build(&g, params),
        .kimivl => try mtmd.models.kimivl.build(&g, params),
        .paddleocr => try mtmd.models.paddleocr.build(&g, params),
        .kimik25, .kimik31 => try mtmd.models.kimik25.build(&g, params),
        .deepseek4v, .deepseek41v => try mtmd.models.deepseek4v.build(&g, params),
        .cogvlm => try mtmd.models.cogvlm.build(&g, params),
        .hunyuanvl => try mtmd.models.hunyuanvl.build(&g, params),
        .mlp, .mlp_norm, .ldp, .ldpv2, .glm_edge => try mtmd.models.llava.build(&g, params),
        .deepseekocr => try mtmd.models.deepseekocr.build(&g, params),
        .deepseekocr2 => try mtmd.models.deepseekocr2.build(&g, params),
        .lfm2a, .d1omni_a => try mtmd.models.conformer.build(&g, params),
        .gemma4a => try mtmd.models.gemma4a.build(&g, params),
        .gemma4ua => try mtmd.models.gemma4ua.build(&g, params),
        .granite_speech => try mtmd.models.granite_speech.build(&g, params),
        .glm4v, .glm5v => try mtmd.models.glm4v.build(&g, params),
        .qwen3a => try mtmd.models.qwen3a.build(&g, params),
        .mimo_audio => try mtmd.models.mimo_audio.build(&g, params),
        .qwen3tts_spkenc => try mtmd.models.qwen3tts_spkenc.build(&g, params),
        .pockettts_spkenc => try mtmd.models.pockettts_spkenc.build(&g, params),
        .chatterbox_spkenc => try mtmd.models.chatterbox_spkenc.build(&g, params),
        .chatterbox => try mtmd.models.chatterbox_gen.build(&g, params),
        .pockettts_gen => try mtmd.models.pockettts_gen.build(&g, params),
        .qwen3tts_gen => try mtmd.models.qwen3tts_gen.build(&g, params),
        .youtuvl => try mtmd.models.youtuvl.build(&g, params),
        .yasa2 => try mtmd.models.yasa2.build(&g, params),
        .parakeet => try mtmd.models.parakeet.build(&g, params),
        .nemotron_transcribe => try mtmd.models.nemotron_transcribe.build(&g, params),
        .granite4_vision => try mtmd.models.granite4_vision.build(&g, params),
        .lfm3a => try mtmd.models.lfm3a.build(&g, params),
        .soprano => try mtmd.models.soprano.build(&g, params),
        .sensenova_u1 => try mtmd.models.sensenova_u1.build(&g, params),
        .lfm3vl => try mtmd.models.lfm3vl.build(&g, params),
        .voxtral_rt => try mtmd.models.voxtral_rt.build(&g, params),
        .muse_spark_audio => try mtmd.models.muse_spark_audio.build(&g, params),
        .lfm3a_gen => try mtmd.models.lfm3a_gen.build(&g, params),
        .drama3_gen => try mtmd.models.drama3_gen.build(&g, params),
        else => mtmd.impl.abort("mtmd-graph.cpp", 1056, "missing cgraph builder"),
    };
    return .{ .gf = gf, .support_batch = supportBatch(ctx.projType()).?, .ctx0 = g.ctx0 };
}

/// Ports `c2w_state_slot` (mtmd/models/models.h:498 @b1b770d90): a persisted
/// state slot of a gen-audio decoder or a streaming audio encoder.
pub const StateSlot = struct {
    name: [:0]const u8,
    ne0: i64,
    ne1: i64,
    /// graph writes the slot directly, no state_in/state_out copy
    in_place: bool = false,
};

/// Ports `list_gen_state_slots` (mtmd-subcontext.cpp:389 @b1b770d90): the
/// persisted state slots of the gen-audio decoders. The caller frees the
/// slice with `freeSlots`.
pub fn genStateSlots(hparams: *const mtmd.model.Hparams, model: *const mtmd.model.Model) mtmd.clip.Error![]StateSlot {
    return switch (model.proj_type) {
        .qwen3tts_gen => mtmd.models.qwen3tts_gen.listStateSlots(hparams, model),
        .drama3_gen => mtmd.models.drama3_gen.listStateSlots(hparams, model),
        .pockettts_gen => mtmd.models.pockettts_gen.listStateSlots(hparams, model),
        .lfm3a_gen => mtmd.models.lfm3a_gen.listStateSlots(hparams, model),
        else => &.{},
    };
}

/// Ports `list_enc_state_slots` (mtmd-subcontext.cpp:400 @b1b770d90): the
/// state slots of the streaming audio encoders. The caller frees the slice
/// with `freeSlots`.
pub fn encStateSlots(hparams: *const mtmd.model.Hparams, model: *const mtmd.model.Model) mtmd.clip.Error![]StateSlot {
    return switch (model.proj_type) {
        .lfm3a => mtmd.models.lfm3a.listStateSlots(hparams, model),
        .gemma4a => mtmd.models.gemma4a.listStateSlots(hparams, model),
        else => &.{},
    };
}

pub fn freeSlots(slots: []StateSlot) void {
    if (slots.len == 0) return;
    for (slots) |s| std.heap.c_allocator.free(s.name);
    std.heap.c_allocator.free(slots);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
