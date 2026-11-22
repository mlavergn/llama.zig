//! Which builder an architecture uses: `llama_model_mapping`'s choice of
//! class, for the architectures the port has reached.
//!
//! # Provenance
//!
//! Ports the class mapping in `llama.cpp/src/llama-model.cpp` (v1.0.0,
//! `b1b770d90`): upstream picks a `llama_model` subclass per architecture;
//! here a class is a file under this directory exporting
//! `loadArchHparams`, `loadArchTensors` and `buildArchGraph`, and `forArch`
//! is the mapping.

const std = @import("std");
const llama = @import("../module.zig");

/// The type of `get_rope_params` and `get_rope_params_indexer`.
pub const RopeParamsFn = *const fn (model: *const llama.model.Model, cparams: *const llama.cparams.Cparams, il: u32) llama.model.RopeParams;

/// An architecture's three overrides, and the two optional ones,
/// `get_rope_params` and `get_rope_params_indexer` (llama-model.h:994, 997
/// @b1b770d90): a null one is `llama_model`'s own.
pub const Entry = struct {
    loadArchHparams: *const fn (self: *llama.model.Model, ml: *const llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void,
    loadArchTensors: *const fn (self: *llama.model.Model, ml: *llama.loader.Loader, exc: *llama.impl.Exception) llama.impl.Thrown!void,
    buildArchGraph: *const fn (model: *const llama.model.Model, b: *const llama.graph.Builder, params: *const llama.graph.Params) void,
    getRopeParams: ?RopeParamsFn = null,
    getRopeParamsIndexer: ?RopeParamsFn = null,
};

/// An entry from a file's declarations: the three overrides, and
/// `getRopeParams` or `getRopeParamsIndexer` where the class overrides one.
fn entryOf(comptime M: type) Entry {
    return .{
        .loadArchHparams = M.loadArchHparams,
        .loadArchTensors = M.loadArchTensors,
        .buildArchGraph = M.buildArchGraph,
        .getRopeParams = if (@hasDecl(M, "getRopeParams")) M.getRopeParams else null,
        .getRopeParamsIndexer = if (@hasDecl(M, "getRopeParamsIndexer")) M.getRopeParamsIndexer else null,
    };
}

/// `llama_model_mapping`'s choice of class, for the architectures ported.
///
/// Parameters:
/// - `arch`: the architecture the file names.
///
/// Return: its overrides, or null when the port has none.
pub fn forArch(arch: llama.arch_table.Arch) ?Entry {
    return switch (arch) {
        .afmoe => entryOf(llama.models.afmoe),
        .apertus => entryOf(llama.models.apertus),
        .arcee => entryOf(llama.models.arcee),
        .arctic => entryOf(llama.models.arctic),
        .arwkv7 => entryOf(llama.models.arwkv7),
        .baichuan => entryOf(llama.models.baichuan),
        .bailingmoe => entryOf(llama.models.bailingmoe),
        .bailingmoe2 => entryOf(llama.models.bailingmoe2),
        .bailingmoe3 => entryOf(llama.models.bailingmoe3),
        .beam => entryOf(llama.models.beam),
        .bert => entryOf(llama.models.bert),
        .bitnet => entryOf(llama.models.bitnet),
        .bloom => entryOf(llama.models.bloom),
        .chameleon => entryOf(llama.models.chameleon),
        .chatglm => entryOf(llama.models.chatglm),
        .clef => entryOf(llama.models.clef),
        .clip => entryOf(llama.models.clip),
        .clover_moe => entryOf(llama.models.clover_moe),
        .codeshell => entryOf(llama.models.codeshell),
        .cogvlm => entryOf(llama.models.cogvlm),
        .cohere2 => entryOf(llama.models.cohere2),
        .cohere2moe => entryOf(llama.models.cohere2moe),
        .command_r => entryOf(llama.models.command_r),
        .dbrx => entryOf(llama.models.dbrx),
        .deci => entryOf(llama.models.deci),
        .deepseek => entryOf(llama.models.deepseek),
        .deepseek2 => entryOf(llama.models.deepseek2),
        .deepseek2ocr => entryOf(llama.models.deepseek2ocr),
        .deepseek32 => entryOf(llama.models.deepseek32),
        .deepseek4 => entryOf(llama.models.deepseek4),
        .deepseek41 => entryOf(llama.models.deepseek41),
        .dflash => entryOf(llama.models.dflash),
        .dots1 => entryOf(llama.models.dots1),
        .dots3note => entryOf(llama.models.dots3note),
        .dream => entryOf(llama.models.dream),
        .eagle3 => entryOf(llama.models.eagle3),
        .ernie4_5 => entryOf(llama.models.ernie4_5),
        .ernie4_5_moe => .{ .loadArchHparams = llama.models.ernie4_5.loadArchHparams, .loadArchTensors = llama.models.ernie4_5.loadArchTensors, .buildArchGraph = llama.models.ernie4_5_moe.buildArchGraph },
        .eurobert => entryOf(llama.models.eurobert),
        .exaone => entryOf(llama.models.exaone),
        .exaone4 => entryOf(llama.models.exaone4),
        .exaone_moe => entryOf(llama.models.exaone_moe),
        .falcon => entryOf(llama.models.falcon),
        .falcon_h1 => entryOf(llama.models.falcon_h1),
        .flint => entryOf(llama.models.flint),
        .gemma => entryOf(llama.models.gemma),
        .gemma2 => entryOf(llama.models.gemma2),
        .gemma3 => entryOf(llama.models.gemma3),
        .gemma3n => entryOf(llama.models.gemma3n),
        .gemma4 => entryOf(llama.models.gemma4),
        .gemma4_assistant => entryOf(llama.models.gemma4_assistant),
        .gemma_embedding => entryOf(llama.models.gemma_embedding),
        .gemma_embedding2 => entryOf(llama.models.gemma_embedding2),
        .glm4 => entryOf(llama.models.glm4),
        .glm4_moe => entryOf(llama.models.glm4_moe),
        .glm5_next => entryOf(llama.models.glm5_next),
        .glm_dsa => entryOf(llama.models.glm_dsa),
        .gpt2 => entryOf(llama.models.gpt2),
        .gptneox => entryOf(llama.models.gptneox),
        .granite => entryOf(llama.models.granite),
        .granite_hybrid => entryOf(llama.models.granite_hybrid),
        .granite_moe => entryOf(llama.models.granite_moe),
        .granite_swa => entryOf(llama.models.granite_swa),
        .granite_switch => entryOf(llama.models.granite_switch),
        .grok => entryOf(llama.models.grok),
        .grovemoe => entryOf(llama.models.grovemoe),
        .hrm_text => entryOf(llama.models.hrm_text),
        .hunyuan_dense => .{ .loadArchHparams = llama.models.hunyuan_vl.loadArchHparams, .loadArchTensors = llama.models.hunyuan_vl.loadArchTensors, .buildArchGraph = llama.models.hunyuan_dense.buildArchGraph, .getRopeParams = llama.models.hunyuan_dense.getRopeParams },
        .hunyuan_moe => entryOf(llama.models.hunyuan_moe),
        .hunyuan_vl => entryOf(llama.models.hunyuan_vl),
        .hy_v3 => entryOf(llama.models.hy_v3),
        .hy_v4 => entryOf(llama.models.hy_v4),
        .internlm2 => entryOf(llama.models.internlm2),
        .jais => entryOf(llama.models.jais),
        .jais2 => entryOf(llama.models.jais2),
        .jamba => entryOf(llama.models.jamba),
        .jina_bert_v2 => entryOf(llama.models.jina_bert_v2),
        .jina_bert_v3 => entryOf(llama.models.jina_bert_v3),
        .k2_horizon => entryOf(llama.models.k2_horizon),
        .kimi_k3 => entryOf(llama.models.kimi_k3),
        .kimi_k31 => entryOf(llama.models.kimi_k31),
        .kimi_linear => entryOf(llama.models.kimi_linear),
        .kolibri1 => entryOf(llama.models.kolibri1),
        .laguna => entryOf(llama.models.laguna),
        .lfm2 => entryOf(llama.models.lfm2),
        .lfm2moe => entryOf(llama.models.lfm2moe),
        .lfm3 => entryOf(llama.models.lfm3),
        .limite => entryOf(llama.models.limite),
        .llada => entryOf(llama.models.llada),
        .llada_moe => entryOf(llama.models.llada_moe),
        .llama => entryOf(llama.models.llama),
        .llama4 => entryOf(llama.models.llama4),
        .llama_embed => .{ .loadArchHparams = llama.models.llama.loadArchHparams, .loadArchTensors = llama.models.llama.loadArchTensors, .buildArchGraph = llama.models.llama.buildArchGraphEmbed },
        .maincoder => entryOf(llama.models.maincoder),
        .maion_coder => entryOf(llama.models.maion_coder),
        .mamba => entryOf(llama.models.mamba),
        .mamba2 => entryOf(llama.models.mamba2),
        .maple => entryOf(llama.models.maple),
        .mellum => entryOf(llama.models.mellum),
        .mimo2 => entryOf(llama.models.mimo2),
        .minicpm => entryOf(llama.models.minicpm),
        .minicpm3 => entryOf(llama.models.minicpm3),
        .minimax_01 => entryOf(llama.models.minimax_01),
        .minimax_m2 => entryOf(llama.models.minimax_m2),
        .minimax_m3 => entryOf(llama.models.minimax_m3),
        .metis16 => entryOf(llama.models.metis16),
        .mistral3 => entryOf(llama.models.mistral3),
        .mistral4 => .{ .loadArchHparams = llama.models.deepseek2.loadArchHparams, .loadArchTensors = llama.models.deepseek2.loadArchTensors, .buildArchGraph = llama.models.mistral4.buildArchGraph },
        .mistral_large4 => entryOf(llama.models.mistral_large4),
        .modern_bert => entryOf(llama.models.modern_bert),
        .mpt => entryOf(llama.models.mpt),
        .muse_glimmer => entryOf(llama.models.muse_glimmer),
        .muse_spark => entryOf(llama.models.muse_spark),
        .nanbeige => entryOf(llama.models.nanbeige),
        .nemotron => entryOf(llama.models.nemotron),
        .nemotron_h => entryOf(llama.models.nemotron_h),
        .nemotron_h_moe => .{ .loadArchHparams = llama.models.nemotron_h.loadArchHparams, .loadArchTensors = llama.models.nemotron_h.loadArchTensors, .buildArchGraph = llama.models.nemotron_h_moe.buildArchGraph },
        .neo_bert => entryOf(llama.models.neo_bert),
        .nomic_bert => entryOf(llama.models.nomic_bert),
        .nomic_bert_moe => entryOf(llama.models.nomic_bert_moe),
        .olmo => entryOf(llama.models.olmo),
        .olmo2 => entryOf(llama.models.olmo2),
        .olmoe => entryOf(llama.models.olmoe),
        .openai_moe => entryOf(llama.models.openai_moe),
        .openelm => entryOf(llama.models.openelm),
        .orion => entryOf(llama.models.orion),
        .paddleocr => .{ .loadArchHparams = llama.models.ernie4_5.loadArchHparams, .loadArchTensors = llama.models.ernie4_5.loadArchTensors, .buildArchGraph = llama.models.paddleocr.buildArchGraph },
        .pangu_embed => entryOf(llama.models.pangu_embed),
        .phi2 => entryOf(llama.models.phi2),
        .phi3 => entryOf(llama.models.phi3),
        .phimoe => entryOf(llama.models.phimoe),
        .plamo => entryOf(llama.models.plamo),
        .plamo2 => entryOf(llama.models.plamo2),
        .plamo3 => entryOf(llama.models.plamo3),
        .plm => entryOf(llama.models.plm),
        .pockettts => entryOf(llama.models.pockettts),
        .qwen => entryOf(llama.models.qwen),
        .qwen2 => entryOf(llama.models.qwen2),
        .qwen2moe => entryOf(llama.models.qwen2moe),
        .qwen2vl => entryOf(llama.models.qwen2vl),
        .qwen3 => entryOf(llama.models.qwen3),
        .qwen35 => entryOf(llama.models.qwen35),
        .qwen35moe => entryOf(llama.models.qwen35moe),
        .qwen3moe => entryOf(llama.models.qwen3moe),
        .qwen3next => entryOf(llama.models.qwen3next),
        .qwen3tts => entryOf(llama.models.qwen3vl),
        .qwen3vl => entryOf(llama.models.qwen3vl),
        .qwen3vlmoe => entryOf(llama.models.qwen3vlmoe),
        .qwen4exp => entryOf(llama.models.qwen4exp),
        .refact => entryOf(llama.models.refact),
        .ricursive => entryOf(llama.models.ricursive),
        .rnd1 => entryOf(llama.models.rnd1),
        .rwkv6 => entryOf(llama.models.rwkv6),
        .rwkv6qwen2 => entryOf(llama.models.rwkv6qwen2),
        .rwkv7 => entryOf(llama.models.rwkv7),
        .seed_oss => entryOf(llama.models.seed_oss),
        .sensenova_u1 => entryOf(llama.models.sensenova_u1),
        .smallthinker => entryOf(llama.models.smallthinker),
        .smollm3 => entryOf(llama.models.smollm3),
        .spark2_5 => entryOf(llama.models.spark2_5),
        .stablelm => entryOf(llama.models.stablelm),
        .starcoder => entryOf(llama.models.starcoder),
        .starcoder2 => entryOf(llama.models.starcoder2),
        .step35 => entryOf(llama.models.step35),
        .t5 => entryOf(llama.models.t5),
        .t5encoder => entryOf(llama.models.t5encoder),
        .talkie => entryOf(llama.models.talkie),
        .vllm_decision2 => entryOf(llama.models.vllm_decision2),
        .wavtokenizer_dec => entryOf(llama.models.wavtokenizer_dec),
        .xing4_0 => entryOf(llama.models.xing4_0),
        .xverse => entryOf(llama.models.xverse),
        else => null,
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an architecture with no entry is unported" {
    try std.testing.expect(forArch(.qwen35) != null);
    try std.testing.expect(forArch(.unknown) == null);
}
