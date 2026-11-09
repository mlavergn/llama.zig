// Upstream's server data layer behind a record driver, for scripts/server-task-diff.
//
// The reference half of the oracle for `tools/server/`'s data layer
// (`common.zig`, `tokens.zig`, `oaicompat.zig`, `task.zig`, `result.zig`,
// `schema.zig`): it compiles beside upstream's `server-common.cpp`,
// `server-task.cpp`, `server-schema.cpp` and `server-chat.cpp`, unmodified,
// and runs each input record through them, writing one transcript;
// `harness/server_task_diff.zig` writes the same transcript through the
// port. Both log through `common_log` at debug verbosity, with a marker per
// record, so the logs are compared too.
//
// Input: one record per line, `<op> <payload>`, the payload JSON on one
// line. The ops are documented beside `run` and in scripts/server-task-diff.
//
// Usage: server_task_ref <records> <transcript> <text-model> <mmproj> <vocab>...

#include "server-common.h"
#include "server-task.h"
#include "server-schema.h"
#include "server-chat.h"
#include "server-decision.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "json.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static std::string out;

static void line(const std::string & s) {
    out += s + "\n";
}

static void str(const char * label, const std::string & s) {
    out += label;
    out += "(" + std::to_string(s.size()) + "):" + s + "\n";
}

// an exception's text on one line
static std::string one_line(const std::string & s) {
    std::string r;
    for (char ch : s) {
        if (ch == '\n') r += "\\n"; else r += ch;
    }
    return r;
}

// Every exception the server's ex_wrapper tells apart; any other
// std::exception -- the Jinja engine's raised and rethrown ones -- reads
// as a runtime_error, which ex_wrapper answers alike.
template <typename F> static void guarded(F f) {
    try {
        f();
    } catch (const common_json_error & e) {
        str("json_error", one_line(e.what()));
    } catch (const std::invalid_argument & e) {
        str("invalid_argument", one_line(e.what()));
    } catch (const std::exception & e) {
        str("runtime_error", one_line(e.what()));
    }
}

static std::string hex_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    char b[16];
    snprintf(b, sizeof(b), "%08x", u);
    return b;
}

static uint64_t fnv(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= ((const uint8_t *) p)[i];
        h *= 1099511628211ull;
    }
    return h;
}

static raw_buffer unhex(const std::string & hex) {
    raw_buffer b;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) b.push_back((uint8_t) std::stoi(hex.substr(i, 2), nullptr, 16));
    return b;
}

static std::string toks_str(const llama_tokens & t) {
    std::string s = "[";
    for (size_t i = 0; i < t.size(); i++) {
        if (i) s += ",";
        s += std::to_string(t[i]);
    }
    return s + "]";
}

// the loaded models
static std::vector<llama_model *> vocab_models;
static llama_model * text_model = nullptr;
static llama_context * text_ctx = nullptr;
static mtmd_context * mctx = nullptr;
static common_params params_base;

static const llama_vocab * vocab_at(const json & j) {
    int i = j.value("v", 0);
    return llama_model_get_vocab(vocab_models.at(i));
}

static std::vector<llama_logit_bias> eog_bias(const llama_vocab * vocab) {
    std::vector<llama_logit_bias> res;
    const int n = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < n; i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            res.push_back({i, -INFINITY});
        }
    }
    return res;
}

// everything task_params holds that to_json leaves out
static json params_extra(const task_params & p) {
    json lists = json::object();
    lists["reasoning_budget_start"] = p.sampling.reasoning_budget_start;
    json ends = json::array();
    for (const auto & e : p.sampling.reasoning_budget_end) ends.push_back(e);
    lists["reasoning_budget_end"] = ends;
    lists["reasoning_budget_forced"] = p.sampling.reasoning_budget_forced;
    return json {
        {"cache_prompt", p.cache_prompt},
        {"include_usage", p.include_usage},
        {"return_tokens", p.return_tokens},
        {"return_progress", p.return_progress},
        {"sse_ping_interval", p.sse_ping_interval},
        {"n_indent", p.n_indent},
        {"n_cmpl", p.n_cmpl},
        {"n_cache_reuse", p.n_cache_reuse},
        {"t_max_predict_ms", p.t_max_predict_ms},
        {"verbose", p.verbose},
        {"response_fields", p.response_fields},
        {"grammar_type", (int) p.sampling.grammar.type},
        {"sampling_generation_prompt", p.sampling.generation_prompt},
        {"reasoning_control", p.sampling.reasoning_control},
        {"reasoning_budget_tokens", p.sampling.reasoning_budget_tokens},
        {"lists", lists},
        {"chat_format_int", (int) p.chat_format},
        {"reasoning_format_int", (int) p.reasoning_format},
        {"n_message_spans", p.message_spans.spans.size()},
        {"n_logit_bias", p.sampling.logit_bias.size()},
    };
}

static task_params eval_params(const json & j) {
    const llama_vocab * vocab = vocab_at(j);
    auto eog = eog_bias(vocab);
    return server_schema::eval_llama_cmpl_schema(vocab, params_base, j.value("eog", true) ? eog : std::vector<llama_logit_bias>(), j.at("data"));
}

// ---------------------------------------------------------------------------
// results

static std::map<int, std::unique_ptr<task_result_state>> states;

static void fill_stats(server_slot_stats & s, const json & j) {
    s.n_prompt_cached     = j.value("cached", 0);
    s.n_prompt_processed  = j.value("processed", 0);
    s.n_gen               = j.value("gen", 0);
    s.n_draft_tokens      = j.value("draft", 0);
    s.n_draft_accepted    = j.value("accepted", 0);
    s.t_start             = j.value("t_start", (int64_t) 0);
    s.t_prompt_last       = j.value("t_prompt", (int64_t) 0);
    s.t_gen_last          = j.value("t_gen", (int64_t) 0);
}

static std::vector<completion_token_output> probs_of(const json & j) {
    std::vector<completion_token_output> res;
    for (const auto & p : j) {
        completion_token_output o;
        o.tok = p.value("tok", 0);
        o.prob = p.value("prob", 0.0f);
        o.text_to_send = p.value("text", std::string());
        for (const auto & q : p.value("probs", json::array())) {
            o.probs.push_back({q.value("tok", 0), q.value("txt", std::string()), q.value("prob", 0.0f)});
        }
        res.push_back(o);
    }
    return res;
}

static llama_tokens tokens_of(const json & j, const char * key) {
    return j.contains(key) ? j.at(key).get<std::vector<int>>() : llama_tokens();
}

// a result's generated text: plain, or tokenized over the record's
// vocabulary (common_chat_input_tokenize) when it says "tokenize"
static common_chat_input content_of(const json & j, const json & f) {
    const std::string text = f.value("content", std::string());
    if (f.value("tokenize", false)) {
        common_chat_input in = common_chat_input_tokenize(vocab_at(j), text);
        std::string toks;
        for (auto t : in.tokens) toks += std::to_string(t) + " ";
        line("ctoks " + toks);
        return in;
    }
    return common_chat_input(text);
}

// what a chat session holds, and what it applies to a task's sampling
static json session_json(const common_chat_session & s) {
    json parts = json::array();
    for (const auto & p : s.prompt().parts) parts.push_back(json::array({p.text, p.is_input}));
    common_params_sampling sp;
    s.apply_sampling(sp);
    json triggers = json::array();
    for (const auto & t : sp.grammar_triggers) triggers.push_back(json::array({(int) t.type, t.value, t.token}));
    return json {
        {"has_template", s.has_template()},
        {"format", common_chat_format_name(s.format())},
        {"parts", parts},
        {"grammar", s.grammar()},
        {"generation_prompt", s.generation_prompt()},
        {"thinking_start_tag", s.thinking_start_tag()},
        {"thinking_end_tags", s.thinking_end_tags()},
        {"additional_stops", s.additional_stops()},
        {"msg", json::array({s.msg().role, s.msg().content, s.msg().reasoning_content})},
        {"parser_empty", s.parser().empty()},
        {"sp.grammar", json::array({(int) sp.grammar.type, sp.grammar.grammar})},
        {"sp.grammar_lazy", sp.grammar_lazy},
        {"sp.generation_prompt", sp.generation_prompt},
        {"sp.preserved_tokens", std::vector<int>(sp.preserved_tokens.begin(), sp.preserved_tokens.end())},
        {"sp.grammar_triggers", triggers},
    };
}

static void run_result(const json & j) {
    const std::string kind = j.at("kind").get<std::string>();
    const json f = j.value("f", json::object());
    std::unique_ptr<server_task_result> r;
    if (kind == "final") {
        auto * x = new server_task_result_cmpl_final();
        x->content = content_of(j, f);
        x->tokens = tokens_of(f, "tokens");
        x->stream = f.value("stream", false);
        x->include_usage = f.value("include_usage", false);
        fill_stats(x->stats, f.value("stats", json::object()));
        x->prompt = f.value("prompt", std::string());
        x->truncated = f.value("truncated", false);
        x->n_decoded = f.value("n_decoded", 0);
        x->n_prompt_tokens = f.value("n_prompt_tokens", 0);
        x->n_prompt_tokens_cache = f.value("n_prompt_tokens_cache", 0);
        x->n_tokens_cached = f.value("n_tokens_cached", 0);
        x->has_new_line = f.value("has_new_line", false);
        x->stopping_word = f.value("stopping_word", std::string());
        x->stop = (stop_type) f.value("stop", 0);
        x->post_sampling_probs = f.value("post_sampling_probs", false);
        x->probs_output = probs_of(f.value("probs", json::array()));
        x->response_fields = f.value("response_fields", std::vector<std::string>());
        if (j.contains("gp")) {
            x->generation_params = eval_params(j.at("gp"));
        }
        x->verbose = f.value("verbose", false);
        x->res_type = (task_response_type) f.value("res_type", 0);
        x->oaicompat_model = f.value("model", std::string());
        x->oaicompat_cmpl_id = f.value("cmpl_id", std::string());
        r.reset(x);
    } else if (kind == "partial") {
        auto * x = new server_task_result_cmpl_partial();
        x->content = content_of(j, f);
        x->tokens = tokens_of(f, "tokens");
        x->n_decoded = f.value("n_decoded", 0);
        x->n_prompt_tokens = f.value("n_prompt_tokens", 0);
        x->n_prompt_tokens_cache = f.value("n_prompt_tokens_cache", 0);
        x->post_sampling_probs = f.value("post_sampling_probs", false);
        x->is_progress = f.value("is_progress", false);
        x->is_begin = f.value("is_begin", false);
        auto po = probs_of(f.value("probs", json::array()));
        if (!po.empty()) x->prob_output = po[0];
        fill_stats(x->stats, f.value("stats", json::object()));
        const json pr = f.value("progress", json::object());
        x->progress.total = pr.value("total", 0);
        x->progress.cache = pr.value("cache", 0);
        x->progress.processed = pr.value("processed", 0);
        x->progress.time_ms = pr.value("time_ms", (int64_t) 0);
        x->verbose = f.value("verbose", false);
        x->res_type = (task_response_type) f.value("res_type", 0);
        x->oaicompat_model = f.value("model", std::string());
        x->oaicompat_cmpl_id = f.value("cmpl_id", std::string());
        r.reset(x);
    } else if (kind == "embd") {
        auto * x = new server_task_result_embd();
        for (const auto & e : f.value("embedding", json::array())) x->embedding.push_back(e.get<std::vector<float>>());
        x->n_tokens = f.value("n_tokens", 0);
        x->res_type = (task_response_type) f.value("res_type", 0);
        r.reset(x);
    } else if (kind == "realtime") {
        auto * x = new server_task_result_realtime();
        x->text = f.value("text", std::string());
        x->audio = f.value("audio", std::vector<float>());
        x->response_done = f.value("response_done", false);
        x->cancelled = f.value("cancelled", false);
        x->n_decoded = f.value("n_decoded", 0);
        r.reset(x);
    } else if (kind == "tts") {
        auto * x = new server_task_result_tts();
        x->audio = f.value("audio", std::string());
        x->sample_rate = f.value("sample_rate", 0);
        x->final = f.value("final", false);
        r.reset(x);
    } else if (kind == "rerank") {
        auto * x = new server_task_result_rerank();
        if (f.contains("score")) x->score = f.value("score", 0.0f);
        x->n_tokens = f.value("n_tokens", 0);
        r.reset(x);
    } else if (kind == "error") {
        auto * x = new server_task_result_error();
        x->err_type = (error_type) f.value("type", 2);
        x->err_msg = f.value("msg", std::string());
        x->n_prompt_tokens = f.value("n_prompt_tokens", 0);
        x->n_ctx = f.value("n_ctx", 0);
        r.reset(x);
    } else if (kind == "metrics") {
        auto * x = new server_task_result_metrics();
        x->n_processing_slots = f.value("processing", 0);
        x->n_tasks_deferred = f.value("deferred", 0);
        auto & m = x->metrics;
        auto bucket = [&](server_metrics::bucket & b, const char * k) {
            const json bj = f.value(k, json::array({0, 0, 0}));
            b.count = bj.at(0).get<unsigned long long>();
            b.steps = bj.at(1).get<unsigned long long>();
            b.time  = bj.at(2).get<unsigned long long>();
        };
        bucket(m.prompt_bucket, "prompt_bucket");
        bucket(m.predict_bucket, "predict_bucket");
        bucket(m.prompt, "prompt");
        bucket(m.predict, "predict");
        m.n_prompt_cached = f.value("n_prompt_cached", 0ull);
        m.n_tokens_max = f.value("n_tokens_max", 0ull);
        m.n_decode = f.value("n_decode", 0ull);
        m.n_busy_slots = f.value("n_busy_slots", 0ull);
        m.n_draft_tokens = f.value("n_draft_tokens", 0ull);
        m.n_draft_accepted = f.value("n_draft_accepted", 0ull);
        m.n_draft_verif_steps = f.value("n_draft_verif_steps", 0ull);
        for (const auto & v : f.value("per_pos", json::array())) m.n_accepted_per_pos.push_back(v.get<unsigned long long>());
        r.reset(x);
        str("metrics", x->to_metrics());
    } else if (kind == "slots") {
        auto * x = new server_task_result_slots();
        x->slots_data = f.value("data", json::array());
        r.reset(x);
    } else if (kind == "save") {
        auto * x = new server_task_result_slot_save_load();
        x->filename = f.value("filename", std::string());
        x->is_save = f.value("is_save", false);
        x->n_tokens = f.value("n_tokens", (size_t) 0);
        x->n_bytes = f.value("n_bytes", (size_t) 0);
        x->t_ms = f.value("t_ms", 0.0);
        r.reset(x);
    } else if (kind == "erase") {
        auto * x = new server_task_result_slot_erase();
        x->n_erased = f.value("n_erased", (size_t) 0);
        r.reset(x);
    } else if (kind == "control") {
        auto * x = new server_task_result_control();
        x->success = f.value("success", false);
        x->message = f.value("message", std::string());
        r.reset(x);
    } else if (kind == "get_lora") {
        auto * x = new server_task_result_get_lora();
        for (const auto & l : f.value("loras", json::array())) {
            server_task_result_get_lora::lora e;
            e.info.path = l.value("path", std::string());
            e.info.scale = l.value("scale", 0.0f);
            e.info.task_name = l.value("task_name", std::string());
            e.info.prompt_prefix = l.value("prompt_prefix", std::string());
            e.info.ptr = nullptr;
            e.alora_invocation_string = l.value("inv", std::string());
            e.alora_invocation_tokens = tokens_of(l, "inv_tokens");
            x->loras.push_back(e);
        }
        r.reset(x);
    } else if (kind == "apply_lora") {
        r.reset(new server_task_result_apply_lora());
    } else if (kind == "router") {
        auto * x = new server_task_result_router();
        x->data = f.value("data", json());
        r.reset(x);
        std::unique_ptr<server_task_result> copy(x->clone());
        str("router_clone", copy->to_json().dump());
    } else {
        throw std::runtime_error("bad kind");
    }
    r->id = j.value("id", -1);
    r->id_slot = j.value("id_slot", -1);
    r->index = j.value("index", (size_t) 0);
    line(std::string("is_error ") + (r->is_error() ? "1" : "0") + " is_stop " + (r->is_stop() ? "1" : "0"));
    const int st = j.value("state", -1);
    if (st >= 0) {
        guarded([&] { r->update(*states.at(st)); });
        const auto & s = *states.at(st);
        str("state.content", s.chat_msg.content);
        line(std::string("state.flags ") + (s.thinking_block_started ? "1" : "0") + (s.text_block_started ? "1" : "0") + (s.oai_resp_created ? "1" : "0"));
    }
    guarded([&] {
        json v = r->to_json();
        str("json", v.dump());
        str("sse_oai", format_oai_sse(v));
        str("sse_anthropic", format_anthropic_sse(v));
        guarded([&] { str("sse_resp", format_oai_resp_sse(v)); });
    });
}

// ---------------------------------------------------------------------------
// server_tokens

static void show_tokens(const char * label, const server_tokens & t) {
    str(label, t.str());
}

static void tokens_ops(server_tokens & a, const json & ops) {
    for (const auto & op : ops) {
        const std::string o = op.at(0).get<std::string>();
        guarded([&] {
            if (o == "pos_next") {
                // the arguments are clamped to the size: past it is GGML_ASSERT
                const long long n = std::min(op.at(1).get<long long>(), (long long) a.size());
                line("pos_next " + std::to_string(n) + " " + std::to_string(a.pos_next(n)));
            } else if (o == "size_up_to_pos") {
                line("size_up_to_pos " + std::to_string(a.size_up_to_pos(op.at(1).get<int>())));
            } else if (o == "keep_first") {
                server_tokens b = a.clone();
                b.keep_first(std::min(op.at(1).get<size_t>(), b.size()));
                show_tokens("kept", b);
            } else if (o == "validate") {
                line(std::string("validate ") + (a.validate(text_ctx) ? "1" : "0"));
            } else if (o == "text") {
                line("text " + toks_str(a.get_text_tokens()));
            } else if (o == "detok") {
                // only tokens of the text model's vocabulary: libllama throws out_of_range past it
                bool ok = true;
                for (auto t : a.get_text_tokens()) ok = ok && t >= 0 && t < llama_vocab_n_tokens(llama_model_get_vocab(text_model));
                if (ok) str("detok", a.detokenize(text_ctx, op.at(1).get<bool>())); else line("detok skipped");
            } else if (o == "serialize") {
                std::vector<char> s = a.serialize();
                line("serialized " + std::to_string(s.size()) + " " + std::to_string(fnv(s.data(), s.size())));
                llama_tokens packed(s.size() / sizeof(llama_token));
                memcpy(packed.data(), s.data(), s.size());
                guarded([&] { show_tokens("restored", server_tokens::deserialize(packed, true)); });
                guarded([&] { show_tokens("restored_nomtmd", server_tokens::deserialize(packed, false)); });
                if (op.size() > 1) {
                    // a corrupted copy: truncated to n tokens, one token overwritten
                    llama_tokens bad = packed;
                    const size_t n = op.at(1).get<size_t>();
                    bad.resize(n, 0);
                    if (op.size() > 3 && op.at(2).get<size_t>() < bad.size()) bad[op.at(2).get<size_t>()] = op.at(3).get<int>();
                    guarded([&] { show_tokens("restored_bad", server_tokens::deserialize(bad, a.has_mtmd)); });
                }
            } else if (o == "clone") {
                show_tokens("clone", a.clone());
            } else if (o == "prefix") {
                const json & other = op.at(1);
                server_tokens b(other.get<std::vector<int>>(), a.has_mtmd);
                line("prefix " + std::to_string(a.get_common_prefix(b)));
            } else if (o == "prefix_self") {
                server_tokens b = a.clone();
                if (op.size() > 1) b.keep_first(std::min(op.at(1).get<size_t>(), b.size()));
                line("prefix_self " + std::to_string(a.get_common_prefix(b)) + " " + std::to_string(b.get_common_prefix(a)));
            } else if (o == "prefix_edit") {
                // a copy cut at n and continued with another token
                server_tokens b = a.clone();
                b.keep_first(std::min(op.at(1).get<size_t>(), b.size()));
                b.push_back((llama_token) op.at(2).get<int>());
                line("prefix_edit " + std::to_string(a.get_common_prefix(b)) + " " + std::to_string(b.get_common_prefix(a)));
            } else if (o == "next_media") {
                auto res = a.find_next_media_chunk(op.at(1).get<size_t>());
                line("next_media " + std::string(res.first ? "1" : "0") + " " + std::to_string(res.second));
            } else if (o == "spans") {
                common_chat_msg_delimiters d;
                for (const auto & e : op.at(1)) {
                    d.add(common_chat_role_from_string(e.at(0).get<std::string>()), e.at(1).get<std::string>());
                }
                d.tokenize(llama_model_get_vocab(text_model));
                auto spans = a.find_message_spans(d);
                std::string s = "spans";
                for (const auto & sp : spans.spans) s += " " + std::to_string((int) sp.role) + ":" + std::to_string(sp.pos) + ":" + std::to_string(sp.len);
                line(s);
            } else if (o == "push_back") {
                server_tokens b = a.clone();
                server_tokens c = a.clone();
                b.push_back(c);
                show_tokens("pushed", b);
            } else if (o == "push_token") {
                server_tokens b = a.clone();
                b.push_back((llama_token) op.at(1).get<int>());
                show_tokens("pushed_token", b);
            } else if (o == "set_token") {
                server_tokens b = a.clone();
                if (b.size() > 0 && !b.has_mtmd) {
                    b.set_token(std::min(op.at(1).get<int>(), (int) b.size() - 1), op.at(2).get<int>());
                }
                show_tokens("set", b);
            } else if (o == "find_chunk") {
                const auto & ch = a.find_chunk(op.at(1).get<size_t>());
                line("find_chunk " + std::to_string(mtmd_input_chunk_get_n_tokens(ch.get())));
            } else if (o == "insert") {
                server_tokens b = a.clone();
                b.insert(op.at(1).get<std::vector<int>>());
                show_tokens("inserted", b);
            } else {
                throw std::runtime_error("bad op " + o);
            }
        });
    }
}

// ---------------------------------------------------------------------------

static mtmd_helper_init_opt init_opt;

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// the templates a record names: a template file's text, or a name or
// source given inline, over the record's vocabulary
static common_chat_templates_ptr make_tmpls(const json & j) {
    const std::string tmpl = j.contains("tmpl_file") ? read_file(j.at("tmpl_file").get<std::string>()) : j.value("tmpl", std::string());
    return common_chat_templates_init(vocab_models.at(j.value("v", 0)), tmpl);
}

static server_chat_params make_chat_opt(const json & j) {
    const json & o = j.at("opt");
    server_chat_params opt;
    opt.use_jinja = o.value("jinja", false);
    opt.prefill_assistant = o.value("prefill_assistant", true);
    opt.reasoning_format = (common_reasoning_format) o.value("reasoning_format", 0);
    json kwargs = o.value("kwargs", json::object());
    for (const auto & item : kwargs.items()) {
        opt.chat_template_kwargs[item.key()] = item.value().get<std::string>();
    }
    opt.tmpls = make_tmpls(j);
    opt.allow_image = o.value("allow_image", false);
    opt.allow_audio = o.value("allow_audio", false);
    opt.allow_video = o.value("allow_video", false);
    opt.enable_thinking = o.value("enable_thinking", true);
    opt.reasoning_budget = o.value("reasoning_budget", -1);
    opt.reasoning_budget_message = o.value("reasoning_budget_message", std::string());
    opt.media_path = o.value("media_path", std::string());
    opt.force_pure_content = o.value("force_pure_content", false);
    return opt;
}

static void run(const std::string & op, const std::string & payload) {
    json j;
    try {
        j = json::parse(payload);
    } catch (const common_json_error & e) {
        str("bad_input", e.what());
        return;
    }

    if (op == "B") {
        // params_base members
        if (j.contains("n_parallel")) params_base.n_parallel = j.at("n_parallel").get<int>();
        if (j.contains("n_predict")) params_base.n_predict = j.at("n_predict").get<int>();
        if (j.contains("n_keep")) params_base.n_keep = j.at("n_keep").get<int>();
        if (j.contains("n_cache_reuse")) params_base.n_cache_reuse = j.at("n_cache_reuse").get<int>();
        if (j.contains("cache_prompt")) params_base.cache_prompt = j.at("cache_prompt").get<bool>();
        if (j.contains("antiprompt")) params_base.antiprompt = j.at("antiprompt").get<std::vector<std::string>>();
        if (j.contains("sse_ping_interval")) params_base.sse_ping_interval = j.at("sse_ping_interval").get<int>();
        if (j.contains("verbosity")) params_base.verbosity = j.at("verbosity").get<int>();
        if (j.contains("reasoning_format")) params_base.reasoning_format = (common_reasoning_format) j.at("reasoning_format").get<int>();
        if (j.contains("seed")) params_base.sampling.seed = j.at("seed").get<unsigned int>();
        if (j.contains("temp")) params_base.sampling.temp = j.at("temp").get<float>();
        if (j.contains("dry_base")) params_base.sampling.dry_base = j.at("dry_base").get<float>();
        if (j.contains("samplers")) params_base.sampling.samplers = common_sampler_types_from_chars(j.at("samplers").get<std::string>());
        if (j.contains("spec_types")) {
            params_base.speculative.types.clear();
            for (const auto & t : j.at("spec_types")) params_base.speculative.types.push_back((common_speculative_type) t.get<int>());
        }
        line("base ok");
    } else if (op == "P") {
        guarded([&] {
            task_params p = eval_params(j);
            guarded([&] { str("params", p.to_json(false).dump()); });
            guarded([&] { str("metrics", p.to_json(true).dump()); });
            str("extra", params_extra(p).dump());
        });
    } else if (op == "O") {
        guarded([&] { str("oaicmpl", oaicompat_completion_params_parse(j).dump()); });
    } else if (op == "C") {
        guarded([&] {
            server_chat_params opt = make_chat_opt(j);
            json body = j.at("body");
            std::vector<raw_buffer> files;
            common_chat_session session;
            guarded([&] { str("chat", oaicompat_chat_params_parse(vocab_at(j), body, opt, files, session).dump()); });
            str("session", session_json(session).dump());
            str("body", body.dump());
            for (const auto & f : files) {
                line("file " + std::to_string(f.size()) + " " + std::to_string(fnv(f.data(), f.size())));
            }
        });
    } else if (op == "CS") {
        // a chat request converted, its parameters evaluated, and a result
        // state made from them: the state then parses the R records' text
        // through the request's own parser
        guarded([&] {
            server_chat_params opt = make_chat_opt(j);
            json body = j.at("body");
            std::vector<raw_buffer> files;
            common_chat_session session;
            json parsed = oaicompat_chat_params_parse(vocab_at(j), body, opt, files, session);
            str("chat", parsed.dump());
            str("session", session_json(session).dump());
            json pj = {{"v", j.value("v", 0)}, {"data", parsed}};
            server_task task(SERVER_TASK_TYPE_COMPLETION);
            task.params = eval_params(pj);
            task.tokens = server_tokens(common_tokenize(vocab_at(j), parsed.at("prompt").get<std::string>(), true, true), false);
            task.apply_chat_session(session);
            str("extra", params_extra(task.params).dump());
            str("params", task.params.to_json(false).dump());
            states[j.at("id").get<int>()] = std::make_unique<task_result_state>(session);
            const auto & st = *states[j.at("id").get<int>()];
            str("state.init", st.chat_msg.content);
            str("state.role", st.chat_msg.role);
        });
    } else if (op == "RTOOLS") {
        guarded([&] { str("conv", server_chat_convert_responses_tools(j.at("tools")).dump()); });
    } else if (op == "TS") {
        guarded([&] {
            server_task t(SERVER_TASK_TYPE_TTS);
            server_schema::eval_tts_schema(params_base, t, j);
            str("tts_params", json{
                {"top_k", t.tts_inp.data.top_k}, {"top_p", t.tts_inp.data.top_p}, {"seed", t.tts_inp.data.seed},
                {"n_predict", t.params.n_predict}, {"repeat", t.params.sampling.penalty_repeat},
                {"stream", t.params.stream}, {"sampling_seed", t.params.sampling.seed},
            }.dump());
        });
    } else if (op == "RC") {
        guarded([&] { str("conv", server_chat_convert_responses_to_chatcmpl(j).dump()); });
    } else if (op == "AC") {
        guarded([&] { str("conv", server_chat_convert_anthropic_to_oai(j).dump()); });
    } else if (op == "TC") {
        guarded([&] {
            common_chat_templates_ptr tmpls = make_tmpls(j);
            std::map<std::string, uploaded_file> in_files;
            if (j.contains("file")) {
                in_files[j.value("file_key", std::string("file"))].data = unhex(j.at("file").get<std::string>());
            }
            std::vector<raw_buffer> files;
            files.push_back(raw_buffer{1, 2, 3}); // cleared by the conversion
            guarded([&] { str("conv", convert_transcriptions_to_chatcmpl(j.at("body"), tmpls.get(), in_files, files).dump()); });
            for (const auto & f : files) {
                line("file " + std::to_string(f.size()) + " " + std::to_string(fnv(f.data(), f.size())));
            }
        });
    } else if (op == "T") {
        guarded([&] {
            const llama_vocab * vocab = vocab_at(j);
            auto res = tokenize_input_prompts(vocab, j.value("mtmd", false) ? mctx : nullptr, j.at("prompt"),
                                              j.value("add_special", true), j.value("parse_special", true), init_opt);
            line("n_prompts " + std::to_string(res.size()));
            for (auto & t : res) {
                show_tokens("st", t);
                if (j.contains("ops")) tokens_ops(t, j.at("ops"));
            }
        });
        guarded([&] { line("mixed " + toks_str(tokenize_mixed(vocab_at(j), j.at("prompt"), j.value("add_special", true), j.value("parse_special", true)))); });
    } else if (op == "M") {
        guarded([&] {
            std::vector<raw_buffer> files;
            for (const auto & f : j.value("files", json::array())) {
                files.push_back(unhex(f.get<std::string>()));
            }
            server_tokens t = process_mtmd_prompt(mctx, j.at("prompt").get<std::string>(), files, init_opt, j.value("ph", false));
            show_tokens("mtmd", t);
            if (j.contains("ops")) tokens_ops(t, j.at("ops"));
        });
    } else if (op == "ST") {
        guarded([&] {
            server_tokens t(j.at("tokens").get<std::vector<int>>(), false);
            show_tokens("st", t);
            tokens_ops(t, j.value("ops", json::array()));
        });
    } else if (op == "U") {
        std::string res;
        for (const auto & h : j) {
            const std::string hex = h.get<std::string>();
            std::string s;
            for (size_t i = 0; i + 1 < hex.size(); i += 2) s += (char) std::stoi(hex.substr(i, 2), nullptr, 16);
            res += std::to_string(validate_utf8(s)) + (is_valid_utf8(s) ? "v" : "x") + " ";
        }
        line("utf8 " + res);
    } else if (op == "E") {
        guarded([&] {
            const error_type t = (error_type) j.at("type").get<int>();
            str("error", format_error_response(j.value("msg", std::string()), t).dump());
        });
    } else if (op == "GT") {
        guarded([&] {
            server_grammar_trigger t(j);
            str("trigger", t.to_json().dump());
        });
    } else if (op == "S") {
        // a state over the default (untemplated) session: plain content
        guarded([&] {
            states[j.at("id").get<int>()] = std::make_unique<task_result_state>();
            const auto & s = *states[j.at("id").get<int>()];
            str("state.init", s.chat_msg.content);
            str("state.role", s.chat_msg.role);
            line("state.ids " + std::to_string(s.oai_resp_id.size()) + " " + std::to_string(s.oai_resp_reasoning_id.size()) + " " + std::to_string(s.oai_resp_message_id.size()));
        });
    } else if (op == "R") {
        guarded([&] { run_result(j); });
    } else if (op == "F") {
        guarded([&] {
            json emb = j.at("emb");
            str("embd", format_embeddings_response_oaicompat(j.at("req"), j.value("model", std::string()), emb, j.value("b64", false)).dump());
        });
    } else if (op == "K") {
        guarded([&] {
            std::vector<std::string> texts = j.value("texts", std::vector<std::string>());
            str("rerank", format_response_rerank(j.at("req"), j.value("model", std::string()), j.at("ranks"), j.value("tei", false), texts, j.value("top_n", 0)).dump());
        });
    } else if (op == "I") {
        guarded([&] {
            auto toks = format_prompt_infill(vocab_at(j), j.at("prefix"), j.at("suffix"), j.at("extra"),
                                             j.value("n_batch", 512), j.value("n_predict", -1), j.value("n_ctx", 4096),
                                             j.value("spm", false), j.value("prompt", std::vector<int>()));
            line("infill " + toks_str(toks));
        });
    } else if (op == "Q") {
        guarded([&] {
            std::vector<raw_buffer> qf;
            std::vector<raw_buffer> df;
            for (int i = 0; i < j.value("n_qf", 0); i++) {
                handle_media(qf, j.at("media").get<std::string>(), "");
            }
            for (int i = 0; i < j.value("n_df", 0); i++) {
                handle_media(df, j.at("media").get<std::string>(), "");
            }
            auto t = format_prompt_rerank(vocab_models.at(j.value("v", 0)), vocab_at(j), j.value("mtmd", false) ? mctx : nullptr,
                                          j.at("query").get<std::string>(), j.at("doc").get<std::string>(), qf, df, init_opt);
            show_tokens("rerank_prompt", t);
        });
    } else if (op == "RT") {
        guarded([&] {
            str("rerank_template", format_rerank_template(j.at("tmpl").get<std::string>(), j.at("query").get<std::string>(), j.at("doc").get<std::string>()));
        });
    } else if (op == "G") {
        guarded([&] {
            const llama_vocab * vocab = llama_model_get_vocab(text_model);
            llama_tokens toks = common_tokenize(vocab, j.at("prompt").get<std::string>(), true, true);
            llama_memory_clear(llama_get_memory(text_ctx), true);
            common_batch batch(text_ctx);
            for (size_t i = 0; i < toks.size(); i++) batch.add(toks[i], (llama_pos) i, 0, true);
            if (llama_process(text_ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) throw std::runtime_error("decode failed");
            for (const auto & nt : j.at("n_top")) {
                const int idx = j.value("idx", -1);
                auto cur = get_token_probabilities(text_ctx, idx, nt.get<size_t>());
                std::string s = "probs n=" + std::to_string(cur.size());
                float sum = 0.0f;
                for (size_t i = 0; i < cur.size(); i++) {
                    sum += cur[i].p;
                    if (i < 12) s += " " + std::to_string(cur[i].id) + ":" + hex_bits(cur[i].logit) + ":" + hex_bits(cur[i].p);
                }
                s += " sum=" + hex_bits(sum);
                line(s);
            }
            // the prompt's own tokens: the fixture vocabulary throws out_of_range on some others
            for (auto t : toks) str("fmt", tokens_to_output_formatted_string(text_ctx, t));
            str("fmt", tokens_to_output_formatted_string(text_ctx, LLAMA_TOKEN_NULL));
            str("tokens_to_str", tokens_to_str(text_ctx, toks));
        });
    } else if (op == "PC") {
        guarded([&] {
            server_prompt_cache cache(j.value("limit_mib", 0), j.value("limit_tokens", (size_t) 0));
            server_prompt cur;
            for (const auto & o : j.at("ops")) {
                const std::string k = o.at(0).get<std::string>();
                guarded([&] {
                    if (k == "alloc") {
                        server_prompt p;
                        p.tokens = server_tokens(o.at(1).get<std::vector<int>>(), false);
                        if (o.size() > 4) {
                            for (const auto & sz : o.at(4)) {
                                common_prompt_checkpoint ck;
                                ck.n_tokens = 0;
                                ck.pos_min = 0;
                                ck.pos_max = 0;
                                ck.data_tgt.resize(sz.get<size_t>());
                                p.checkpoints.push_back(ck);
                            }
                        }
                        auto * st = cache.alloc(p, o.at(2).get<size_t>(), o.at(3).get<size_t>());
                        line(std::string("alloc ") + (st ? "1" : "0"));
                    } else if (k == "update") {
                        cache.update();
                        line("update");
                    } else if (k == "cur") {
                        cur.tokens = server_tokens(o.at(1).get<std::vector<int>>(), false);
                        line("cur");
                    } else if (k == "load") {
                        server_tokens nt(o.at(1).get<std::vector<int>>(), false);
                        const bool ok = cache.load(cur, nt, text_ctx, nullptr, 0);
                        line(std::string("load ") + (ok ? "1" : "0") + " " + toks_str(cur.tokens.get_text_tokens()));
                    }
                    std::string s = "cache " + std::to_string(cache.states.size()) + " " + std::to_string(cache.size()) + " " + std::to_string(cache.n_tokens()) + " " + std::to_string(cache.limit_size);
                    for (const auto & st : cache.states) {
                        s += " [" + std::to_string(st.prompt.tokens.size()) + "," + std::to_string(st.data.main.size()) + "," + std::to_string(st.data.drft.size()) + "," + std::to_string(st.prompt.checkpoints.size()) + "]";
                    }
                    line(s);
                });
            }
        });
    } else if (op == "PIPE") {
        server_pipe<int> p;
        p.max_size = j.value("max_size", (size_t) 0);
        std::string s = "pipe";
        for (const auto & o : j.at("ops")) {
            const std::string k = o.at(0).get<std::string>();
            if (k == "w") {
                s += std::string(" w") + (p.write(o.at(1).get<int>()) ? "1" : "0");
            } else if (k == "r") {
                int v = -1;
                const bool ok = p.read(v, [] { return true; }, o.size() > 1 ? o.at(1).get<bool>() : true);
                s += " r" + std::string(ok ? "1:" : "0:") + std::to_string(v);
            } else if (k == "cw") {
                p.close_write();
                s += " cw";
            } else if (k == "cr") {
                p.close_read();
                s += " cr";
            }
        }
        line(s);
    } else if (op == "L") {
        guarded([&] {
            json req = j.at("req");
            auto m = parse_lora_request(req);
            std::string s = "lora";
            for (const auto & e : m) s += " " + std::to_string(e.first) + ":" + hex_bits(e.second);
            line(s);
        });
        if (j.contains("a")) {
            std::vector<common_adapter_lora_info> la, lb;
            for (const auto & sc : j.at("a")) { common_adapter_lora_info i; i.scale = sc.get<float>(); i.ptr = nullptr; la.push_back(i); }
            for (const auto & sc : j.at("b")) { common_adapter_lora_info i; i.scale = sc.get<float>(); i.ptr = nullptr; lb.push_back(i); }
            std::string s = std::string("lora_eq ") + (are_lora_equal(la, lb) ? "1" : "0") + " ids";
            for (auto id : lora_get_enabled_ids(la)) s += " " + std::to_string(id);
            line(s);
        }
    } else if (op == "J") {
        guarded([&] {
            const json & js = j.at("js");
            str("nested", json_get_nested_values(j.at("paths").get<std::vector<std::string>>(), js).dump());
            line(std::string("is ") + (json_is_array_of_numbers(js) ? "1" : "0") + (json_is_array_of_mixed_numbers_strings(js) ? "1" : "0") + (json_is_array_and_contains_numbers(js) ? "1" : "0"));
        });
    } else if (op == "SSE") {
        guarded([&] { str("sse_oai", format_oai_sse(j)); });
        guarded([&] { str("sse_anthropic", format_anthropic_sse(j)); });
        guarded([&] { str("sse_resp", format_oai_resp_sse(j)); });
    } else if (op == "RND") {
        const std::string r = random_string();
        bool ok = r.size() == 32;
        for (char ch : r) ok = ok && isalnum((unsigned char) ch);
        const std::string c = gen_chatcmplid();
        line(std::string("random ") + (ok ? "1" : "0") + " " + c.substr(0, 9) + " " + std::to_string(c.size()) + " " + std::to_string(gen_tool_call_id().size()) + " " + (r != random_string() ? "1" : "0"));
        str("marker", get_media_marker());
    } else if (op == "TK") {
        server_task t((server_task_type) j.value("type", 0));
        t.params.sampling.seed = j.value("seed", (unsigned int) LLAMA_DEFAULT_SEED);
        t.id = 10;
        for (int i = 0; i < j.value("children", 0); i++) t.add_child(10, 11 + i);
        std::string s = "task";
        s += std::string(" embd") + (t.need_embd() ? "1" : "0") + " logits" + (t.need_logits() ? "1" : "0") + " sampling" + (t.need_sampling() ? "1" : "0");
        s += " parent" + std::string(t.is_parent() ? "1" : "0");
        for (auto & ch : t.child_tasks) s += " " + std::to_string(ch.id) + ":" + std::to_string(ch.id_parent) + ":" + std::to_string(ch.params.sampling.seed) + ":" + (ch.is_child() ? "c" : "-");
        std::vector<server_task> v;
        v.push_back(std::move(t));
        s += " ids " + std::to_string(server_task::get_list_id(v).size());
        line(s);
    } else if (op == "D") {
        // a /v1/systemone request through a decision model's context (one of
        // the vocabularies, loaded vocab-only, carries the metadata): the
        // questions and the state parsed, every task filled, the tasks
        // grouped, and each answer formatted from scores taken in order from
        // the record's pool
        guarded([&] {
            const llama_model * model = vocab_models.at(j.value("v", 0));
            line("decision_type " + std::to_string((int) common_get_decision_type(model)));
            server_decision_context dc;
            dc.init(model);
            line(std::string("flags share") + (dc.can_share_prompt() ? "1" : "0") + " joint" + (dc.is_joint() ? "1" : "0") +
                 " images" + (dc.can_use_images() ? "1" : "0") + " generate" + (dc.can_generate() ? "1" : "0") + " lfm3" + (dc.is_lfm3_d1() ? "1" : "0") +
                 " type" + std::to_string((int) dc.type));
            if (dc.type == COMMON_DECISION_TYPE_NONE) {
                return;
            }
            const json body = j.at("body");
            const auto questions = dc.parse_questions(body);
            for (const auto & q : questions) {
                std::string s = "q " + q.id + " " + std::to_string((int) q.type) + " " + q.instructions.dump();
                for (const auto & o : q.options) s += " [" + o.key + "|" + o.description.dump() + "]";
                str("question", s);
                line("variants " + std::to_string(dc.n_variants(q)));
            }
            std::vector<raw_buffer> files;
            const json state = dc.prepare_state(dc.parse_state(body, files));
            str("state", state.dump());
            for (const auto & f : files) {
                line("file " + std::to_string(f.size()) + " " + std::to_string(fnv(f.data(), f.size())));
            }
            // the model is vocab-only: a request with files would tokenize them through mtmd
            if (!files.empty()) {
                return;
            }
            std::vector<server_task> tasks;
            auto show_task = [&](const server_task & t) {
                std::string s = "task " + std::to_string(t.id) + " labels" + toks_str(t.decision.labels) + " markers" + toks_str(t.decision.markers) +
                                " column" + std::to_string(t.decision.column) + " pointer" + std::to_string(t.decision.pointer) +
                                " first" + std::to_string(t.decision.pos_first()) + " order" + toks_str(t.decision.order) +
                                " n_scores" + std::to_string(t.decision.n_scores) +
                                " embd" + (t.need_embd() ? "1" : "0") + " logits" + (t.need_logits() ? "1" : "0") +
                                " no_batching" + (t.decision.no_batching ? "1" : "0") + " pooled" + (t.decision.pooled ? "1" : "0");
                line(s);
                show_tokens("prompt", t.tokens);
            };
            int id = 100;
            if (dc.is_joint()) {
                server_task t(SERVER_TASK_TYPE_DECISION);
                t.id = id++;
                dc.fill_task_joint(state, questions, files, nullptr, init_opt, t);
                show_task(t);
                tasks.push_back(std::move(t));
            } else {
                for (const auto & q : questions) {
                    for (size_t v = 0; v < dc.n_variants(q); v++) {
                        server_task t(SERVER_TASK_TYPE_DECISION);
                        t.id = id++;
                        dc.fill_task(state, questions, q, v, files, nullptr, init_opt, t);
                        show_task(t);
                        tasks.push_back(std::move(t));
                    }
                }
            }
            // the outputs each task reads: its labels, its markers, or for a joint head one per option
            std::vector<size_t> n_out;
            for (const auto & t : tasks) {
                n_out.push_back(!t.decision.labels.empty() ? t.decision.labels.size() : !t.decision.markers.empty() ? t.decision.markers.size() : (size_t) t.decision.n_scores);
            }
            const auto pool = j.value("pool", std::vector<float>());
            size_t i_pool = 0;
            auto take = [&](size_t n) {
                std::vector<float> v;
                for (size_t k = 0; k < n; k++) v.push_back(pool.empty() ? 0.0f : pool[i_pool++ % pool.size()]);
                return v;
            };
            const int nan_at = j.value("nan_at", -1);
            size_t i_task = 0;
            size_t i_score = 0;
            int n_answer = 0;
            for (const auto & q : questions) {
                std::vector<std::vector<float>> scores;
                if (dc.is_joint()) {
                    auto all = take(n_out[0]);
                    scores.emplace_back(all.begin() + std::min(all.size(), i_score), all.begin() + std::min(all.size(), i_score + q.options.size()));
                    i_score += q.options.size();
                } else {
                    for (size_t v = 0; v < dc.n_variants(q); v++) scores.push_back(take(n_out[i_task++]));
                }
                if (n_answer++ == nan_at && !scores.empty() && !scores[0].empty()) scores[0][0] = NAN;
                if (j.value("short", false) && !scores.empty() && !scores[0].empty()) scores[0].pop_back();
                guarded([&] { str("answer", dc.format_answer(q, scores).dump()); });
            }
            if (dc.can_share_prompt()) {
                auto groups = server_decision_group_tasks(std::move(tasks), j.value("n_slots", 1));
                for (const auto & g : groups) {
                    std::string s = "group " + std::to_string(g.id) + " shared" + std::to_string(g.n_tokens_shared) + " n" + std::to_string(g.tokens.size());
                    for (const auto & ch : g.child_tasks) s += " " + std::to_string(ch.id) + ":" + std::to_string(ch.id_parent) + ":" + std::to_string(ch.tokens.size());
                    line(s);
                }
            }
        });
    } else if (op == "DR") {
        guarded([&] {
            server_task_result_decision r;
            r.index = j.value("index", 0);
            r.n_tokens = j.value("n_tokens", 0);
            r.scores = j.value("scores", std::vector<float>());
            str("decision_result", r.to_json().dump());
        });
    } else if (op == "MA") {
        guarded([&] {
            const auto out_mod = server_model_output_modalities((common_decision_type) j.value("type", 0), j.value("chat", true), j.value("tts", false));
            str("arch", server_model_architecture_json(j.value("image", false), j.value("audio", false), j.value("video", false), out_mod).dump());
        });
    } else if (op == "OE") {
        guarded([&] {
            server_chat_params opt = make_chat_opt(j);
            const llama_vocab * vocab = vocab_at(j);
            server_tokens t = tokenize_oai_content_array(vocab, j.value("mtmd", false) ? mctx : nullptr, opt, j.at("content"),
                                                         j.value("add_special", true), j.value("parse_special", true), init_opt);
            show_tokens("oe", t);
        });
        guarded([&] {
            server_tokens t = tokenize_input_subprompt(vocab_at(j), j.value("mtmd", false) ? mctx : nullptr, j.at("content"),
                                                       j.value("add_special", true), j.value("parse_special", true), init_opt);
            show_tokens("sub", t);
        });
    } else {
        str("bad_op", op);
    }
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <records> <transcript> <text-model> <mmproj> <vocab>...\n", argv[0]);
        return 2;
    }
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    mtmd_helper_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    params_base.n_parallel = 4;
    init_opt = mtmd_helper_init_opt_default();

    for (int i = 5; i < argc; i++) {
        auto mp = llama_model_default_params();
        // a decision model's labels and classes are read past where a vocab-only load stops
        const std::string path = argv[i];
        const std::string full = "-full.gguf";
        mp.vocab_only = path.size() < full.size() || path.compare(path.size() - full.size(), full.size(), full) != 0;
        mp.n_gpu_layers = 0;
        llama_model * m = llama_model_load_from_file(argv[i], mp);
        if (!m) {
            fprintf(stderr, "cannot load %s\n", argv[i]);
            return 2;
        }
        vocab_models.push_back(m);
    }
    {
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        text_model = llama_model_load_from_file(argv[3], mp);
        if (!text_model) {
            fprintf(stderr, "cannot load %s\n", argv[3]);
            return 2;
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_threads = 1;
        cp.n_threads_batch = 1;
        text_ctx = llama_init_from_model(text_model, cp);
        auto mpar = mtmd_context_params_default();
        mpar.use_gpu = false;
        mpar.print_timings = false;
        mpar.n_threads = 1;
        mpar.warmup = false;
        mpar.media_marker = get_media_marker();
        mctx = std::string(argv[4]) == "-" ? nullptr : mtmd_init_from_file(argv[4], text_model, mpar);
        if (!text_ctx || (!mctx && std::string(argv[4]) != "-")) {
            fprintf(stderr, "cannot make the contexts\n");
            return 2;
        }
    }

    std::ifstream in(argv[1]);
    std::string ln;
    int n = 0;
    while (std::getline(in, ln)) {
        const size_t sp = ln.find(' ');
        const std::string op = ln.substr(0, sp);
        const std::string payload = sp == std::string::npos ? "" : ln.substr(sp + 1);
        LOG_WRN("@@ %d %s\n", n, op.c_str());
        line("@@ " + std::to_string(n) + " " + op);
        run(op, payload);
        n++;
    }
    common_log_flush(common_log_main());

    std::ofstream f(argv[2], std::ios::binary);
    f << out;
    if (mctx) mtmd_free(mctx);
    llama_free(text_ctx);
    llama_model_free(text_model);
    for (auto * m : vocab_models) llama_model_free(m);
    return 0;
}
