//! The per-model encoder graphs (`llama.cpp/tools/mtmd/models/`), and
//! `mtmd_get_graph_builder` (`llama.cpp/tools/mtmd/mtmd-graph.cpp`, v1.0.0,
//! `b1b770d90`), which picks one by projector type, with
//! `list_gen_state_slots` and `list_enc_state_slots`
//! (`llama.cpp/tools/mtmd/mtmd-subcontext.cpp`), which list the state slots
//! of the builders that keep state between calls.
//!
//! Each file here ports the `models/*.cpp` of the same name (hyphens become
//! underscores) and exposes `build(g, params)`: the class's `build()`, run
//! on a `graph.Graph` set up by `dispatch.zig`'s `buildGraph`. A model that
//! overrides `build_mm` sets `g.mm_hook` first.

const std = @import("std");

pub const chatterbox_base = @import("chatterbox_base.zig");
pub const chatterbox_gen = @import("chatterbox_gen.zig");
pub const chatterbox_spkenc = @import("chatterbox_spkenc.zig");
pub const cogvlm = @import("cogvlm.zig");
pub const conformer = @import("conformer.zig");
pub const deepseek4v = @import("deepseek4v.zig");
pub const deepseekocr = @import("deepseekocr.zig");
pub const deepseekocr2 = @import("deepseekocr2.zig");
pub const dots3note = @import("dots3note.zig");
pub const dotsocr = @import("dotsocr.zig");
pub const drama3_gen = @import("drama3_gen.zig");
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
pub const lfm3a = @import("lfm3a.zig");
pub const lfm3a_gen = @import("lfm3a_gen.zig");
pub const lfm3vl = @import("lfm3vl.zig");
pub const llama4 = @import("llama4.zig");
pub const ling3vl = @import("ling3vl.zig");
pub const llava = @import("llava.zig");
pub const magevl = @import("magevl.zig");
pub const mimo_audio = @import("mimo_audio.zig");
pub const mimovl = @import("mimovl.zig");
pub const minicpmv = @import("minicpmv.zig");
pub const minimax_m3 = @import("minimax_m3.zig");
pub const mobilenetv5 = @import("mobilenetv5.zig");
pub const muse_glimmer = @import("muse_glimmer.zig");
pub const muse_spark_audio = @import("muse_spark_audio.zig");
pub const nemotron_omni = @import("nemotron_omni.zig");
pub const nemotron_transcribe = @import("nemotron_transcribe.zig");
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
pub const sensenova_u1 = @import("sensenova_u1.zig");
pub const siglip = @import("siglip.zig");
pub const soprano = @import("soprano.zig");
pub const step3vl = @import("step3vl.zig");
pub const voxtral_rt = @import("voxtral_rt.zig");
pub const whisper_enc = @import("whisper_enc.zig");
pub const yasa2 = @import("yasa2.zig");
pub const youtuvl = @import("youtuvl.zig");
pub const dispatch = @import("dispatch.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
