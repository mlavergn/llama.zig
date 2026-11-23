//! The architectures: one file per upstream `src/models/*.cpp`, and the
//! table that says which one an `Arch` loads and builds with.
//!
//! **Not a port.** Upstream gives each architecture a class deriving from
//! `llama_model_base` and overriding three virtuals -- `load_arch_hparams`,
//! `load_arch_tensors` and `build_arch_graph` (llama-model.h:1014, 1015, 1016
//! @b1b770d90) -- and `llama_model_mapping` (llama-model.cpp:43 @b1b770d90) picks the
//! class. Here a class is a file exporting those three functions, and
//! `dispatch.zig`'s `forArch` is the mapping. An architecture with no entry is one the port
//! has not reached: loading it stops at `error.Unported`, which
//! `testing/scripts/arch-model-diff` counts.

const std = @import("std");

pub const afmoe = @import("afmoe.zig");
pub const apertus = @import("apertus.zig");
pub const arcee = @import("arcee.zig");
pub const arctic = @import("arctic.zig");
pub const arwkv7 = @import("arwkv7.zig");
pub const baichuan = @import("baichuan.zig");
pub const bailingmoe = @import("bailingmoe.zig");
pub const bailingmoe2 = @import("bailingmoe2.zig");
pub const bailingmoe3 = @import("bailingmoe3.zig");
pub const beam = @import("beam.zig");
pub const bert = @import("bert.zig");
pub const bitnet = @import("bitnet.zig");
pub const bloom = @import("bloom.zig");
pub const chameleon = @import("chameleon.zig");
pub const chatglm = @import("chatglm.zig");
pub const clef = @import("clef.zig");
pub const clip = @import("clip.zig");
pub const clover_moe = @import("clover_moe.zig");
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
pub const deepseek41 = @import("deepseek41.zig");
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
pub const flint = @import("flint.zig");
pub const gemma = @import("gemma.zig");
pub const gemma2 = @import("gemma2.zig");
pub const gemma3 = @import("gemma3.zig");
pub const gemma3n = @import("gemma3n.zig");
pub const gemma4 = @import("gemma4.zig");
pub const gemma4_assistant = @import("gemma4_assistant.zig");
pub const gemma_embedding = @import("gemma_embedding.zig");
pub const gemma_embedding2 = @import("gemma_embedding2.zig");
pub const glm4 = @import("glm4.zig");
pub const glm4_moe = @import("glm4_moe.zig");
pub const glm5_next = @import("glm5_next.zig");
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
pub const hrm_text = @import("hrm_text.zig");
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
pub const k2_horizon = @import("k2_horizon.zig");
pub const kimi_k3 = @import("kimi_k3.zig");
pub const kimi_k31 = @import("kimi_k31.zig");
pub const kimi_linear = @import("kimi_linear.zig");
pub const kolibri1 = @import("kolibri1.zig");
pub const laguna = @import("laguna.zig");
pub const lfm2 = @import("lfm2.zig");
pub const lfm2moe = @import("lfm2moe.zig");
pub const lfm3 = @import("lfm3.zig");
pub const lfm_base = @import("lfm_base.zig");
pub const limite = @import("limite.zig");
pub const llada = @import("llada.zig");
pub const llada_moe = @import("llada_moe.zig");
pub const llama = @import("llama.zig");
pub const llama4 = @import("llama4.zig");
pub const maincoder = @import("maincoder.zig");
pub const maion_coder = @import("maion_coder.zig");
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
pub const metis16 = @import("metis16.zig");
pub const mistral3 = @import("mistral3.zig");
pub const mistral4 = @import("mistral4.zig");
pub const mistral_large4 = @import("mistral_large4.zig");
pub const modern_bert = @import("modern_bert.zig");
pub const mpt = @import("mpt.zig");
pub const muse_glimmer = @import("muse_glimmer.zig");
pub const muse_spark = @import("muse_spark.zig");
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
pub const ricursive = @import("ricursive.zig");
pub const rnd1 = @import("rnd1.zig");
pub const rwkv6 = @import("rwkv6.zig");
pub const rwkv6_base = @import("rwkv6_base.zig");
pub const rwkv6qwen2 = @import("rwkv6qwen2.zig");
pub const rwkv7 = @import("rwkv7.zig");
pub const rwkv7_base = @import("rwkv7_base.zig");
pub const seed_oss = @import("seed_oss.zig");
pub const sensenova_u1 = @import("sensenova_u1.zig");
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
pub const vllm_decision2 = @import("vllm_decision2.zig");
pub const wavtokenizer_dec = @import("wavtokenizer_dec.zig");
pub const xing4_0 = @import("xing4_0.zig");
pub const xverse = @import("xverse.zig");
pub const dispatch = @import("dispatch.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
