// Upstream's chat data model behind a record driver, for scripts/chat-msg-diff.
//
// The reference half of the oracle for `cli/common_chat/msg.zig`: it links
// upstream's `llama-common` (`common/chat.cpp` unmodified) and runs each
// input record through the `common_chat_*` calls the port reproduces,
// writing one transcript; `harness/chat_msg_diff.zig` writes the same
// transcript through the port. The two must be identical.
//
// Input: one record per line, `<op> <payload>`, where the payload is JSON on
// one line. Ops: M messages, T tools, C tool_choice, K continuation,
// D delimiters, S split, F diff, R reasoning format, P tool parameters.

#include "chat.h"
#include "json.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string out;

static void str(const char * label, const std::string & s) {
    out += label;
    out += "(" + std::to_string(s.size()) + "):" + s + "\n";
}

static void line(const std::string & s) {
    out += s + "\n";
}

template <typename F> static void guarded(F f) {
    try {
        f();
    } catch (const common_json_error & e) {
        str("json_error", e.what());
    } catch (const std::invalid_argument & e) {
        str("invalid_argument", e.what());
    } catch (const std::runtime_error & e) {
        str("runtime_error", e.what());
    }
}

static void show_msg(const common_chat_msg & m) {
    str("role", m.role);
    str("content", m.content);
    for (const auto & p : m.content_parts) {
        str("part.type", p.type);
        str("part.text", p.text);
    }
    for (const auto & tc : m.tool_calls) {
        str("tc.name", tc.name);
        str("tc.arguments", tc.arguments);
        str("tc.id", tc.id);
    }
    str("reasoning_content", m.reasoning_content);
    str("tool_name", m.tool_name);
    str("tool_call_id", m.tool_call_id);
    line(std::string("empty ") + (m.empty() ? "1" : "0") + " media " + (m.contains_media() ? "1" : "0"));
    guarded([&] { str("oai", m.to_json_oaicompat(false).dump()); });
    guarded([&] { str("oai_concat", m.to_json_oaicompat(true).dump()); });
    guarded([&] { str("render", m.render_content()); });
    guarded([&] { str("render_bar", m.render_content("|")); });
}

static common_chat_msg one_msg(const common_json & j) {
    common_json arr = common_json::array();
    arr.push_back(j);
    return common_chat_msgs_parse_oaicompat(arr).at(0);
}

static void run(char op, const std::string & payload) {
    common_json j;
    try {
        j = common_json::parse(payload);
    } catch (const common_json_error & e) {
        str("bad_input", e.what());
        return;
    }
    switch (op) {
        case 'M': guarded([&] {
            auto msgs = common_chat_msgs_parse_oaicompat(j);
            line("n " + std::to_string(msgs.size()));
            for (const auto & m : msgs) {
                show_msg(m);
            }
            guarded([&] { str("all", common_chat_msgs_to_json_oaicompat(msgs, false).dump()); });
            guarded([&] { str("all_concat", common_chat_msgs_to_json_oaicompat(msgs, true).dump(2)); });
            for (size_t i = 0; i < msgs.size(); i++) {
                for (size_t k = 0; k < msgs.size(); k++) {
                    line(std::string("eq ") + (msgs[i] == msgs[k] ? "1" : "0"));
                }
            }
        }); break;
        case 'T': guarded([&] {
            auto tools = common_chat_tools_parse_oaicompat(j);
            line("n " + std::to_string(tools.size()));
            for (const auto & t : tools) {
                str("name", t.name);
                str("description", t.description);
                str("parameters", t.parameters);
            }
            guarded([&] { str("back", common_chat_tools_to_json_oaicompat(tools).dump()); });
        }); break;
        case 'C': guarded([&] {
            line("choice " + std::to_string((int) common_chat_tool_choice_parse_oaicompat(j.get<std::string>())));
        }); break;
        case 'K':
            line("continuation " + std::to_string((int) common_chat_continuation_parse(j)));
            break;
        case 'D': guarded([&] {
            auto d = common_chat_msg_delimiters_parse(j);
            line("n " + std::to_string(d.delimiters.size()));
            str("json", d.to_json().dump());
        }); break;
        case 'S': guarded([&] {
            common_chat_msg_delimiters d;
            for (const auto & e : j.at("delims")) {
                common_chat_msg_delimiter x;
                x.role = common_chat_role_from_string(e.at("role").get<std::string>());
                x.delimiter = e.at("delimiter").get<std::string>();
                x.tokens = e.at("tokens").get<std::vector<int>>();
                d.delimiters.push_back(x);
            }
            std::map<size_t, size_t> skips;
            for (const auto & s : j.at("skips")) {
                skips[s.at((size_t) 0).get<size_t>()] = s.at((size_t) 1).get<size_t>();
            }
            auto spans = d.split(j.at("tokens").get<std::vector<int>>(), skips);
            for (const auto & s : spans.spans) {
                line("span " + std::string(common_chat_role_to_string(s.role)) + " " + std::to_string(s.pos) + " " +
                     std::to_string(s.len) + " " + (s.valid() ? "1" : "0"));
            }
            for (int p = -1; p < 12; p++) {
                line("user_start " + std::to_string(p) + " " + (spans.is_user_start(p) ? "1" : "0"));
            }
            line("last_user " + std::to_string(spans.last_user_message_pos()));
        }); break;
        case 'F': guarded([&] {
            auto prv = one_msg(j.at((size_t) 0));
            auto cur = one_msg(j.at((size_t) 1));
            auto diffs = common_chat_msg_diff::compute_diffs(prv, cur);
            line("n " + std::to_string(diffs.size()));
            for (const auto & d : diffs) {
                str("reasoning_delta", d.reasoning_content_delta);
                str("content_delta", d.content_delta);
                line("index " + std::to_string(d.tool_call_index));
                str("delta.name", d.tool_call_delta.name);
                str("delta.arguments", d.tool_call_delta.arguments);
                str("delta.id", d.tool_call_delta.id);
            }
            for (size_t i = 0; i < diffs.size(); i++) {
                for (size_t k = 0; k < diffs.size(); k++) {
                    line(std::string("eq ") + (diffs[i] == diffs[k] ? "1" : "0"));
                }
            }
        }); break;
        case 'R': guarded([&] {
            auto f = common_reasoning_format_from_name(j.get<std::string>());
            line(std::string("format ") + std::to_string((int) f) + " " + common_reasoning_format_name(f));
        }); break;
        case 'P': guarded([&] {
            str("params", common_chat_tool_parameters(j).dump());
        }); break;
        default:
            line("unknown op");
    }
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <in> <out>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::string rec;
    size_t n = 0;
    while (std::getline(in, rec)) {
        if (rec.size() < 2) {
            continue;
        }
        out += "## " + std::to_string(n++) + " " + rec[0] + "\n";
        run(rec[0], rec.substr(2));
    }
    for (int f = 0; f < COMMON_CHAT_FORMAT_COUNT; f++) {
        line(std::string("format_name ") + common_chat_format_name((common_chat_format) f));
    }
    for (const char * r : { "system", "assistant", "user", "tool", "", "User", "developer" }) {
        line(std::string("role ") + r + " " + std::to_string((int) common_chat_role_from_string(r)) + " " +
             common_chat_role_to_string(common_chat_role_from_string(r)));
    }
    std::ofstream o(argv[2], std::ios::binary);
    o << out;
    return 0;
}
