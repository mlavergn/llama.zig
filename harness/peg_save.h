// A PEG arena as JSON, for the gates that compare two arenas
// (scripts/peg-diff, scripts/chat-parse-diff).
//
// Not a port. The form upstream's own `common_peg_arena::save` wrote until
// v0.7.0 removed it, read through the arena's public members -- every
// parser by `get(id)`, the root -- with the rule map, whose libc++ order no
// caller can see any more, sorted by name. `harness/peg_serialize.zig`
// writes the same for the port.

#pragma once

#include "json.h"
#include "peg-parser.h"

#include <map>
#include <string>
#include <type_traits>
#include <variant>

inline common_json harness_peg_variant_json(const common_peg_parser_variant & variant) {
    using json = common_json;
    return std::visit([](const auto & p) -> json {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, common_peg_epsilon_parser>) {
            return json{{"type", "epsilon"}};
        } else if constexpr (std::is_same_v<T, common_peg_start_parser>) {
            return json{{"type", "start"}};
        } else if constexpr (std::is_same_v<T, common_peg_end_parser>) {
            return json{{"type", "end"}};
        } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
            return json{{"type", "literal"}, {"literal", p.literal}};
        } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
            return json{{"type", "sequence"}, {"children", p.children}};
        } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
            return json{{"type", "choice"}, {"children", p.children}};
        } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
            return json{{"type", "repetition"}, {"child", p.child}, {"min_count", p.min_count}, {"max_count", p.max_count}};
        } else if constexpr (std::is_same_v<T, common_peg_and_parser>) {
            return json{{"type", "and"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_not_parser>) {
            return json{{"type", "not"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
            return json{{"type", "any"}};
        } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
            return json{{"type", "space"}};
        } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
            json ranges = json::array();
            for (const auto & range : p.ranges) {
                ranges.push_back({{"start", range.start}, {"end", range.end}});
            }
            return json{{"type", "chars"}, {"pattern", p.pattern}, {"ranges", ranges}, {"negated", p.negated},
                        {"min_count", p.min_count}, {"max_count", p.max_count}};
        } else if constexpr (std::is_same_v<T, common_peg_string_parser>) {
            return json{{"type", "string"}, {"delimiter", std::string(1, p.delimiter)}};
        } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
            return json{{"type", "until"}, {"delimiters", p.delimiters}};
        } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
            return json{{"type", "schema"}, {"child", p.child}, {"name", p.name}, {"raw", p.raw}};
        } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
            return json{{"type", "rule"}, {"name", p.name}, {"child", p.child}, {"trigger", p.trigger}};
        } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
            return json{{"type", "ref"}, {"name", p.name}};
        } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
            return json{{"type", "atomic"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
            return json{{"type", "tag"}, {"child", p.child}, {"tag", p.tag}};
        } else if constexpr (std::is_same_v<T, common_peg_gbnf_parser>) {
            return json{{"type", "gbnf"}, {"child", p.child}, {"grammar", p.grammar}};
        } else if constexpr (std::is_same_v<T, common_peg_ac_parser>) {
            return json{{"type", "ac"}, {"child", p.child}, {"delimiters", p.delimiters}};
        }
    }, variant);
}

inline std::string harness_peg_save(const common_peg_arena & a) {
    auto parsers = common_json::array();
    std::map<std::string, common_peg_parser_id> rules;
    for (size_t id = 0; id < a.size(); id++) {
        const auto & v = a.get(id);
        parsers.push_back(harness_peg_variant_json(v));
        if (const auto * r = std::get_if<common_peg_rule_parser>(&v)) {
            if (a.has_rule(r->name)) {
                rules[r->name] = a.get_rule(r->name);
            }
        }
    }
    // member by member: `common_json` instantiates its `std::map`
    // constructor only for the value types common/ uses
    common_json rules_json = common_json::object();
    for (const auto & [name, id] : rules) {
        rules_json[name] = id;
    }
    return common_json{{"parsers", parsers}, {"rules", rules_json}, {"root", a.root()}}.dump();
}

