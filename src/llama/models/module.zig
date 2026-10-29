//! The architectures: one file per upstream `src/models/*.cpp`, and the
//! table that says which one an `Arch` loads and builds with.
//!
//! **Not a port.** Upstream gives each architecture a class deriving from
//! `llama_model_base` and overriding three virtuals -- `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph` (llama-model.h:767, 768, 769
//! @b29c606e2) -- and `llama_model_mapping` (llama-model.cpp:43 @b29c606e2) picks the
//! class. Here a class is a file exporting those three functions, and
//! `forArch` is the mapping. An architecture with no entry is one the port
//! has not reached: loading it stops at `error.Unported`, which
//! `scripts/arch-model-diff` counts.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const graph = @import("../graph.zig");
const arch_mod = @import("../arch.zig");

const Model = model_mod.Model;
const Exception = impl.Exception;
const Thrown = impl.Thrown;

pub const afmoe = @import("afmoe.zig");
pub const apertus = @import("apertus.zig");
pub const arcee = @import("arcee.zig");
pub const arctic = @import("arctic.zig");
pub const arwkv7 = @import("arwkv7.zig");
pub const baichuan = @import("baichuan.zig");
pub const bailingmoe = @import("bailingmoe.zig");
pub const bailingmoe2 = @import("bailingmoe2.zig");
pub const bailingmoe3 = @import("bailingmoe3.zig");
pub const bert = @import("bert.zig");
pub const bitnet = @import("bitnet.zig");
pub const bloom = @import("bloom.zig");
pub const chameleon = @import("chameleon.zig");
pub const chatglm = @import("chatglm.zig");
pub const clip = @import("clip.zig");
pub const codeshell = @import("codeshell.zig");
pub const cogvlm = @import("cogvlm.zig");
pub const cohere2 = @import("cohere2.zig");
pub const cohere2moe = @import("cohere2moe.zig");
pub const command_r = @import("command_r.zig");
pub const dbrx = @import("dbrx.zig");
pub const deci = @import("deci.zig");
pub const deepseek = @import("deepseek.zig");
pub const deepseek2 = @import("deepseek2.zig");
pub const deepseek2ocr = @import("deepseek2ocr.zig");
pub const deepseek32 = @import("deepseek32.zig");
pub const deepseek4 = @import("deepseek4.zig");
pub const delta_net = @import("delta_net.zig");
pub const dflash = @import("dflash.zig");
pub const dots1 = @import("dots1.zig");
pub const dots3note = @import("dots3note.zig");
pub const dream = @import("dream.zig");
pub const eagle3 = @import("eagle3.zig");
pub const ernie4_5 = @import("ernie4_5.zig");
pub const ernie4_5_moe = @import("ernie4_5_moe.zig");
pub const eurobert = @import("eurobert.zig");
pub const exaone = @import("exaone.zig");
pub const exaone4 = @import("exaone4.zig");
pub const exaone_moe = @import("exaone_moe.zig");
pub const falcon = @import("falcon.zig");
pub const falcon_h1 = @import("falcon_h1.zig");
pub const gemma = @import("gemma.zig");
pub const gemma2 = @import("gemma2.zig");
pub const gemma3 = @import("gemma3.zig");
pub const gemma3n = @import("gemma3n.zig");
pub const gemma4 = @import("gemma4.zig");
pub const gemma4_assistant = @import("gemma4_assistant.zig");
pub const gemma_embedding = @import("gemma_embedding.zig");
pub const glm4 = @import("glm4.zig");
pub const glm4_moe = @import("glm4_moe.zig");
pub const glm_dsa = @import("glm_dsa.zig");
pub const gpt2 = @import("gpt2.zig");
pub const gptneox = @import("gptneox.zig");
pub const granite = @import("granite.zig");
pub const granite_hybrid = @import("granite_hybrid.zig");
pub const granite_moe = @import("granite_moe.zig");
pub const granite_swa = @import("granite_swa.zig");
pub const granite_switch = @import("granite_switch.zig");
pub const grok = @import("grok.zig");
pub const grovemoe = @import("grovemoe.zig");
pub const hunyuan_dense = @import("hunyuan_dense.zig");
pub const hunyuan_moe = @import("hunyuan_moe.zig");
pub const hunyuan_vl = @import("hunyuan_vl.zig");
pub const hy_v3 = @import("hy_v3.zig");
pub const hy_v4 = @import("hy_v4.zig");
pub const internlm2 = @import("internlm2.zig");
pub const jais = @import("jais.zig");
pub const jais2 = @import("jais2.zig");
pub const jamba = @import("jamba.zig");
pub const jina_bert_v2 = @import("jina_bert_v2.zig");
pub const jina_bert_v3 = @import("jina_bert_v3.zig");
pub const kimi_k3 = @import("kimi_k3.zig");
pub const kimi_linear = @import("kimi_linear.zig");
pub const laguna = @import("laguna.zig");
pub const lfm2 = @import("lfm2.zig");
pub const lfm2moe = @import("lfm2moe.zig");
pub const llada = @import("llada.zig");
pub const llada_moe = @import("llada_moe.zig");
pub const llama = @import("llama.zig");
pub const llama4 = @import("llama4.zig");
pub const maincoder = @import("maincoder.zig");
pub const mamba = @import("mamba.zig");
pub const mamba2 = @import("mamba2.zig");
pub const mamba_base = @import("mamba_base.zig");
pub const maple = @import("maple.zig");
pub const mellum = @import("mellum.zig");
pub const mimo2 = @import("mimo2.zig");
pub const minicpm = @import("minicpm.zig");
pub const minicpm3 = @import("minicpm3.zig");
pub const minimax_01 = @import("minimax_01.zig");
pub const minimax_m2 = @import("minimax_m2.zig");
pub const minimax_m3 = @import("minimax_m3.zig");
pub const mistral3 = @import("mistral3.zig");
pub const mistral4 = @import("mistral4.zig");
pub const modern_bert = @import("modern_bert.zig");
pub const mpt = @import("mpt.zig");
pub const muse_glimmer = @import("muse_glimmer.zig");
pub const nanbeige = @import("nanbeige.zig");
pub const nemotron = @import("nemotron.zig");
pub const nemotron_h = @import("nemotron_h.zig");
pub const nemotron_h_moe = @import("nemotron_h_moe.zig");
pub const neo_bert = @import("neo_bert.zig");
pub const nomic_bert = @import("nomic_bert.zig");
pub const nomic_bert_moe = @import("nomic_bert_moe.zig");
pub const olmo = @import("olmo.zig");
pub const olmo2 = @import("olmo2.zig");
pub const olmoe = @import("olmoe.zig");
pub const openai_moe = @import("openai_moe.zig");
pub const openelm = @import("openelm.zig");
pub const orion = @import("orion.zig");
pub const paddleocr = @import("paddleocr.zig");
pub const pangu_embed = @import("pangu_embed.zig");
pub const phi2 = @import("phi2.zig");
pub const phi3 = @import("phi3.zig");
pub const phimoe = @import("phimoe.zig");
pub const plamo = @import("plamo.zig");
pub const plamo2 = @import("plamo2.zig");
pub const plamo3 = @import("plamo3.zig");
pub const plm = @import("plm.zig");
pub const pockettts = @import("pockettts.zig");
pub const qwen = @import("qwen.zig");
pub const qwen2 = @import("qwen2.zig");
pub const qwen2moe = @import("qwen2moe.zig");
pub const qwen2vl = @import("qwen2vl.zig");
pub const qwen3 = @import("qwen3.zig");
pub const qwen35 = @import("qwen35.zig");
pub const qwen35moe = @import("qwen35moe.zig");
pub const qwen3moe = @import("qwen3moe.zig");
pub const qwen3next = @import("qwen3next.zig");
pub const qwen3vl = @import("qwen3vl.zig");
pub const qwen3vlmoe = @import("qwen3vlmoe.zig");
pub const qwen4exp = @import("qwen4exp.zig");
pub const refact = @import("refact.zig");
pub const rnd1 = @import("rnd1.zig");
pub const rwkv6 = @import("rwkv6.zig");
pub const rwkv6_base = @import("rwkv6_base.zig");
pub const rwkv6qwen2 = @import("rwkv6qwen2.zig");
pub const rwkv7 = @import("rwkv7.zig");
pub const rwkv7_base = @import("rwkv7_base.zig");
pub const seed_oss = @import("seed_oss.zig");
pub const smallthinker = @import("smallthinker.zig");
pub const smollm3 = @import("smollm3.zig");
pub const spark2_5 = @import("spark2_5.zig");
pub const stablelm = @import("stablelm.zig");
pub const starcoder = @import("starcoder.zig");
pub const starcoder2 = @import("starcoder2.zig");
pub const step35 = @import("step35.zig");
pub const t5 = @import("t5.zig");
pub const t5encoder = @import("t5encoder.zig");
pub const talkie = @import("talkie.zig");
pub const wavtokenizer_dec = @import("wavtokenizer_dec.zig");
pub const xverse = @import("xverse.zig");

/// An architecture's three overrides.
pub const Entry = struct {
    loadArchHparams: *const fn (self: *Model, ml: *const loader.Loader, exc: *Exception) Thrown!void,
    loadArchTensors: *const fn (self: *Model, ml: *loader.Loader, exc: *Exception) Thrown!void,
    buildArchGraph: *const fn (model: *const Model, b: *const graph.Builder, params: *const graph.Params) void,
};

fn entryOf(comptime M: type) Entry {
    return .{
        .loadArchHparams = M.loadArchHparams,
        .loadArchTensors = M.loadArchTensors,
        .buildArchGraph = M.buildArchGraph,
    };
}

/// `llama_model_mapping`'s choice of class, for the architectures ported.
///
/// Parameters:
/// - `arch`: the architecture the file names.
///
/// Return: its overrides, or null when the port has none.
pub fn forArch(arch: arch_mod.Arch) ?Entry {
    return switch (arch) {
        .afmoe => entryOf(afmoe),
        .apertus => entryOf(apertus),
        .arcee => entryOf(arcee),
        .arctic => entryOf(arctic),
        .arwkv7 => entryOf(arwkv7),
        .baichuan => entryOf(baichuan),
        .bailingmoe => entryOf(bailingmoe),
        .bailingmoe2 => entryOf(bailingmoe2),
        .bailingmoe3 => entryOf(bailingmoe3),
        .bert => entryOf(bert),
        .bitnet => entryOf(bitnet),
        .bloom => entryOf(bloom),
        .chameleon => entryOf(chameleon),
        .chatglm => entryOf(chatglm),
        .clip => entryOf(clip),
        .codeshell => entryOf(codeshell),
        .cogvlm => entryOf(cogvlm),
        .cohere2 => entryOf(cohere2),
        .cohere2moe => entryOf(cohere2moe),
        .command_r => entryOf(command_r),
        .dbrx => entryOf(dbrx),
        .deci => entryOf(deci),
        .deepseek => entryOf(deepseek),
        .deepseek2 => entryOf(deepseek2),
        .deepseek2ocr => entryOf(deepseek2ocr),
        .deepseek32 => entryOf(deepseek32),
        .deepseek4 => entryOf(deepseek4),
        .dflash => entryOf(dflash),
        .dots1 => entryOf(dots1),
        .dots3note => entryOf(dots3note),
        .dream => entryOf(dream),
        .eagle3 => entryOf(eagle3),
        .ernie4_5 => entryOf(ernie4_5),
        .ernie4_5_moe => .{ .loadArchHparams = ernie4_5.loadArchHparams, .loadArchTensors = ernie4_5.loadArchTensors, .buildArchGraph = ernie4_5_moe.buildArchGraph },
        .eurobert => entryOf(eurobert),
        .exaone => entryOf(exaone),
        .exaone4 => entryOf(exaone4),
        .exaone_moe => entryOf(exaone_moe),
        .falcon => entryOf(falcon),
        .falcon_h1 => entryOf(falcon_h1),
        .gemma => entryOf(gemma),
        .gemma2 => entryOf(gemma2),
        .gemma3 => entryOf(gemma3),
        .gemma3n => entryOf(gemma3n),
        .gemma4 => entryOf(gemma4),
        .gemma4_assistant => entryOf(gemma4_assistant),
        .gemma_embedding => entryOf(gemma_embedding),
        .glm4 => entryOf(glm4),
        .glm4_moe => entryOf(glm4_moe),
        .glm_dsa => entryOf(glm_dsa),
        .gpt2 => entryOf(gpt2),
        .gptneox => entryOf(gptneox),
        .granite => entryOf(granite),
        .granite_hybrid => entryOf(granite_hybrid),
        .granite_moe => entryOf(granite_moe),
        .granite_swa => entryOf(granite_swa),
        .granite_switch => entryOf(granite_switch),
        .grok => entryOf(grok),
        .grovemoe => entryOf(grovemoe),
        .hunyuan_dense => .{ .loadArchHparams = hunyuan_vl.loadArchHparams, .loadArchTensors = hunyuan_vl.loadArchTensors, .buildArchGraph = hunyuan_dense.buildArchGraph },
        .hunyuan_moe => entryOf(hunyuan_moe),
        .hunyuan_vl => entryOf(hunyuan_vl),
        .hy_v3 => entryOf(hy_v3),
        .hy_v4 => entryOf(hy_v4),
        .internlm2 => entryOf(internlm2),
        .jais => entryOf(jais),
        .jais2 => entryOf(jais2),
        .jamba => entryOf(jamba),
        .jina_bert_v2 => entryOf(jina_bert_v2),
        .jina_bert_v3 => entryOf(jina_bert_v3),
        .kimi_k3 => entryOf(kimi_k3),
        .kimi_linear => entryOf(kimi_linear),
        .laguna => entryOf(laguna),
        .lfm2 => entryOf(lfm2),
        .lfm2moe => entryOf(lfm2moe),
        .llada => entryOf(llada),
        .llada_moe => entryOf(llada_moe),
        .llama => entryOf(llama),
        .llama4 => entryOf(llama4),
        .llama_embed => .{ .loadArchHparams = llama.loadArchHparams, .loadArchTensors = llama.loadArchTensors, .buildArchGraph = llama.buildArchGraphEmbed },
        .maincoder => entryOf(maincoder),
        .mamba => entryOf(mamba),
        .mamba2 => entryOf(mamba2),
        .maple => entryOf(maple),
        .mellum => entryOf(mellum),
        .mimo2 => entryOf(mimo2),
        .minicpm => entryOf(minicpm),
        .minicpm3 => entryOf(minicpm3),
        .minimax_01 => entryOf(minimax_01),
        .minimax_m2 => entryOf(minimax_m2),
        .minimax_m3 => entryOf(minimax_m3),
        .mistral3 => entryOf(mistral3),
        .mistral4 => .{ .loadArchHparams = deepseek2.loadArchHparams, .loadArchTensors = deepseek2.loadArchTensors, .buildArchGraph = mistral4.buildArchGraph },
        .modern_bert => entryOf(modern_bert),
        .mpt => entryOf(mpt),
        .muse_glimmer => entryOf(muse_glimmer),
        .nanbeige => entryOf(nanbeige),
        .nemotron => entryOf(nemotron),
        .nemotron_h => entryOf(nemotron_h),
        .nemotron_h_moe => .{ .loadArchHparams = nemotron_h.loadArchHparams, .loadArchTensors = nemotron_h.loadArchTensors, .buildArchGraph = nemotron_h_moe.buildArchGraph },
        .neo_bert => entryOf(neo_bert),
        .nomic_bert => entryOf(nomic_bert),
        .nomic_bert_moe => entryOf(nomic_bert_moe),
        .olmo => entryOf(olmo),
        .olmo2 => entryOf(olmo2),
        .olmoe => entryOf(olmoe),
        .openai_moe => entryOf(openai_moe),
        .openelm => entryOf(openelm),
        .orion => entryOf(orion),
        .paddleocr => .{ .loadArchHparams = ernie4_5.loadArchHparams, .loadArchTensors = ernie4_5.loadArchTensors, .buildArchGraph = paddleocr.buildArchGraph },
        .pangu_embed => entryOf(pangu_embed),
        .phi2 => entryOf(phi2),
        .phi3 => entryOf(phi3),
        .phimoe => entryOf(phimoe),
        .plamo => entryOf(plamo),
        .plamo2 => entryOf(plamo2),
        .plamo3 => entryOf(plamo3),
        .plm => entryOf(plm),
        .pockettts => entryOf(pockettts),
        .qwen => entryOf(qwen),
        .qwen2 => entryOf(qwen2),
        .qwen2moe => entryOf(qwen2moe),
        .qwen2vl => entryOf(qwen2vl),
        .qwen3 => entryOf(qwen3),
        .qwen35 => entryOf(qwen35),
        .qwen35moe => entryOf(qwen35moe),
        .qwen3moe => entryOf(qwen3moe),
        .qwen3next => entryOf(qwen3next),
        .qwen3tts => entryOf(qwen3vl),
        .qwen3vl => entryOf(qwen3vl),
        .qwen3vlmoe => entryOf(qwen3vlmoe),
        .qwen4exp => entryOf(qwen4exp),
        .refact => entryOf(refact),
        .rnd1 => entryOf(rnd1),
        .rwkv6 => entryOf(rwkv6),
        .rwkv6qwen2 => entryOf(rwkv6qwen2),
        .rwkv7 => entryOf(rwkv7),
        .seed_oss => entryOf(seed_oss),
        .smallthinker => entryOf(smallthinker),
        .smollm3 => entryOf(smollm3),
        .spark2_5 => entryOf(spark2_5),
        .stablelm => entryOf(stablelm),
        .starcoder => entryOf(starcoder),
        .starcoder2 => entryOf(starcoder2),
        .step35 => entryOf(step35),
        .t5 => entryOf(t5),
        .t5encoder => entryOf(t5encoder),
        .talkie => entryOf(talkie),
        .wavtokenizer_dec => entryOf(wavtokenizer_dec),
        .xverse => entryOf(xverse),
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
