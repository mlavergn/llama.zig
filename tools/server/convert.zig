//! The request conversions into an OpenAI `/chat/completions` body: the
//! Responses API's, the Anthropic Messages API's and the transcriptions
//! API's, each run before `oaicompat.oaicompatChatParamsParse`.
//!
//! # Provenance
//!
//! Ports `server_chat_convert_responses_tools`,
//! `server_chat_convert_responses_to_chatcmpl`,
//! `normalize_anthropic_billing_header`,
//! `server_chat_convert_anthropic_to_oai`,
//! `convert_transcriptions_to_chatcmpl` and `server_chat_get_asr_prompt` from
//! `llama.cpp/tools/server/server-chat.cpp` and
//! `llama.cpp/tools/server/server-chat.h` (llama.cpp v1.0.0, `b1b770d90`);
//! the file's fourth function, `server_chat_msg_diff_to_json_oaicompat`, is
//! `result.zig`'s. Each declaration names what it ports and the line it
//! began at.
//!
//! # Ownership
//!
//! Every function takes an arena: what it returns and every temporary are
//! the arena's, and nothing is freed singly.

const std = @import("std");
const llama = @import("llama");
const mod = @import("module.zig");
const stdlib = @import("stdlib");
const json = @import("json");
const common_chat = @import("common_chat");
const tools_common = @import("tools_common");

/// `j.contains(key) && j.at(key).is_array()`.
fn existsAndIsArray(j: *const json.Value, key: []const u8) bool {
    return j.contains(key) and (j.at(key) catch |err| switch (err) {
        error.OutOfMemory, error.JsonError => unreachable,
    }).isArray();
}

/// `j.contains(key) && j.at(key).is_string()`.
fn existsAndIsString(j: *const json.Value, key: []const u8) bool {
    return j.contains(key) and (j.at(key) catch |err| switch (err) {
        error.OutOfMemory, error.JsonError => unreachable,
    }).isString();
}

/// `j.at(key) == s`: false for anything but that string.
fn atIs(j: *const json.Value, key: []const u8, s: []const u8) common_chat.exception.Error!bool {
    const v = try j.at(key);
    return v.isString() and std.mem.eql(u8, v.string, s);
}

/// `json { {k0, v0}, {k1, v1}, ... }`: an object of the pairs in order, a
/// repeated key keeping its first place.
fn object(a: std.mem.Allocator, pairs: anytype) common_chat.exception.Error!json.Value {
    var o = json.Value.emptyObject();
    inline for (pairs) |p| try o.set(a, p[0], try mod.common.from(a, p[1]));
    return o;
}

/// `json::array({ v })`.
fn arrayOf(a: std.mem.Allocator, v: json.Value) common_chat.exception.Error!json.Value {
    var arr = json.Value.emptyArray();
    try arr.pushBack(a, v);
    return arr;
}

// -----------------------------------------------------------------------------
// the Responses API

/// Ports `server_chat_convert_responses_tools` (server-chat.cpp:6
/// @b1b770d90): OpenAI Responses API tools as Chat Completions API tools,
/// non-function tools skipped.
///
/// Return: owned by `a`.
pub fn responsesTools(a: std.mem.Allocator, tools_in: *const json.Value) common_chat.exception.Error!json.Value {
    if (!tools_in.isArray()) {
        return common_chat.exception.throw(error.InvalidArgument, "'tools' must be an array of objects", .{});
    }
    var chatcmpl_tools = json.Value.emptyArray();
    for (tools_in.array.items) |*resp_tool_in| {
        var resp_tool = try resp_tool_in.clone(a);
        var chatcmpl_tool: json.Value = .null;

        const ty = mod.common.jsonValue(&resp_tool, "type", []const u8, "");
        if (!std.mem.eql(u8, ty, "function")) {
            // Non-function Responses tools have no Chat Completions equivalent.
            const z = mod.common.dupZ(ty);
            defer mod.common.freeZ(z);
            mod.common.srv(.wrn, "server_chat_convert_responses_tools", "unsupported Responses tool type '%s' skipped\n", .{z});
            continue;
        }
        try resp_tool.erase(a, "type");
        try chatcmpl_tool.set(a, "type", try json.Value.initString(a, "function"));

        if (!resp_tool.contains("strict")) {
            try resp_tool.set(a, "strict", .{ .boolean = true });
        }
        try chatcmpl_tool.set(a, "function", resp_tool);
        try chatcmpl_tools.pushBack(a, chatcmpl_tool);
    }
    return chatcmpl_tools;
}

/// Ports `server_chat_convert_responses_to_chatcmpl` (server-chat.cpp:32
/// @b1b770d90): an OpenAI Responses request as a `/chat/completions` one --
/// `instructions` a system message, each input item a message (an output
/// item, a function call or a reasoning item following an assistant
/// message merged into it), function tools unwrapped, `max_output_tokens`
/// and `reasoning.effort` renamed, `text.format` as `response_format`, and
/// the members of an `extra_params` object (llama.cpp's own parameters)
/// merged in.
///
/// Parameters:
/// - `a`: an arena; the result is its.
///
/// Return: the body; `error.InvalidArgument` for a request it cannot read.
pub fn responsesToChatcmpl(a: std.mem.Allocator, response_body: *const json.Value) common_chat.exception.Error!json.Value {
    if (!response_body.contains("input")) {
        return common_chat.exception.throw(error.InvalidArgument, "'input' is required", .{});
    }
    if (mod.common.jsonValue(response_body, "previous_response_id", []const u8, "").len != 0) {
        return common_chat.exception.throw(error.InvalidArgument, "llama.cpp does not support 'previous_response_id'.", .{});
    }

    const input_value = try response_body.at("input");
    var chatcmpl_body = try response_body.clone(a);
    try chatcmpl_body.erase(a, "input");
    var chatcmpl_messages = json.Value.emptyArray();

    if (response_body.contains("instructions")) {
        try chatcmpl_messages.pushBack(a, try object(a, .{
            .{ "role", "system" },
            .{ "content", mod.common.jsonValue(response_body, "instructions", []const u8, "") },
        }));
        try chatcmpl_body.erase(a, "instructions");
    }

    if (input_value.isString()) {
        // #responses_create-input-text_input
        try chatcmpl_messages.pushBack(a, try object(a, .{
            .{ "role", "user" },
            .{ "content", try input_value.clone(a) },
        }));
    } else if (input_value.isArray()) {
        // #responses_create-input-input_item_list
        for (input_value.array.items) |*item_in| {
            var item = try item_in.clone(a);
            const msgs = chatcmpl_messages.array.items;
            const merge_prev = msgs.len != 0 and std.mem.eql(u8, try msgs[msgs.len - 1].value("role", []const u8, ""), "assistant");

            if (existsAndIsString(&item, "content")) {
                // #responses_create-input-input_item_list-input_message-content-text_input
                // Only "Input message" contains item["content"]::string
                // After converting item["content"]::string to item["content"]::array,
                // we can treat "Input message" as sum of "Item-Input message" and "Item-Output message"
                const content = try object(a, .{
                    .{ "text", try (try item.at("content")).clone(a) },
                    .{ "type", "input_text" },
                });
                try item.set(a, "content", try arrayOf(a, content));
            }

            if (existsAndIsArray(&item, "content") and
                existsAndIsString(&item, "role") and
                (try atIs(&item, "role", "user") or
                    try atIs(&item, "role", "system") or
                    try atIs(&item, "role", "developer")))
            {
                // #responses_create-input-input_item_list-item-input_message
                var chatcmpl_content = json.Value.emptyArray();

                for ((try item.at("content")).array.items) |*input_item| {
                    const ty = mod.common.jsonValue(input_item, "type", []const u8, "");

                    if (std.mem.eql(u8, ty, "input_text")) {
                        if (!input_item.contains("text")) {
                            return common_chat.exception.throw(error.InvalidArgument, "'Input text' requires 'text'", .{});
                        }
                        try chatcmpl_content.pushBack(a, try object(a, .{
                            .{ "text", try (try input_item.at("text")).clone(a) },
                            .{ "type", "text" },
                        }));
                    } else if (std.mem.eql(u8, ty, "input_image")) {
                        // While `detail` is marked as required,
                        // it has default value("auto") and can be omitted.

                        if (!input_item.contains("image_url")) {
                            return common_chat.exception.throw(error.InvalidArgument, "'image_url' is required", .{});
                        }
                        try chatcmpl_content.pushBack(a, try object(a, .{
                            .{ "image_url", try object(a, .{.{ "url", try (try input_item.at("image_url")).clone(a) }}) },
                            .{ "type", "image_url" },
                        }));
                    } else if (std.mem.eql(u8, ty, "input_file")) {
                        return common_chat.exception.throw(error.InvalidArgument, "'input_file' is not supported by llamacpp at this moment", .{});
                    } else {
                        return common_chat.exception.throw(error.InvalidArgument, "'type' must be one of 'input_text', 'input_image', or 'input_file'", .{});
                    }
                }

                if (item.contains("type")) {
                    try item.erase(a, "type");
                }
                if (item.contains("status")) {
                    try item.erase(a, "status");
                }
                try item.set(a, "content", chatcmpl_content);

                try chatcmpl_messages.pushBack(a, item);
            } else if (existsAndIsString(&item, "role") and
                try atIs(&item, "role", "assistant") and
                existsAndIsString(&item, "type") and
                try atIs(&item, "type", "message"))
            {
                // #responses_create-input-input_item_list-item-output_message
                var chatcmpl_content = json.Value.emptyArray();

                // Handle both string content and array content
                if (item.contains("content") and (try item.at("content")).isString()) {
                    // String content - convert to text content part
                    try chatcmpl_content.pushBack(a, try object(a, .{
                        .{ "text", try (try item.at("content")).clone(a) },
                        .{ "type", "text" },
                    }));
                } else if (existsAndIsArray(&item, "content")) {
                    // Array content - process each item
                    for ((try item.at("content")).array.items) |*output_text| {
                        const ty = mod.common.jsonValue(output_text, "type", []const u8, "");
                        if (std.mem.eql(u8, ty, "output_text") or std.mem.eql(u8, ty, "input_text")) {
                            // Accept both output_text and input_text (string content gets converted to input_text)
                            if (!existsAndIsString(output_text, "text")) {
                                return common_chat.exception.throw(error.InvalidArgument, "'Output text' requires 'text'", .{});
                            }
                            try chatcmpl_content.pushBack(a, try object(a, .{
                                .{ "text", try (try output_text.at("text")).clone(a) },
                                .{ "type", "text" },
                            }));
                        } else if (std.mem.eql(u8, ty, "refusal")) {
                            if (!existsAndIsString(output_text, "refusal")) {
                                return common_chat.exception.throw(error.InvalidArgument, "'Refusal' requires 'refusal'", .{});
                            }
                            try chatcmpl_content.pushBack(a, try object(a, .{
                                .{ "refusal", try (try output_text.at("refusal")).clone(a) },
                                .{ "type", "refusal" },
                            }));
                        } else {
                            return common_chat.exception.throw(error.InvalidArgument, "'type' must be one of 'output_text' or 'refusal'", .{});
                        }
                    }
                }

                if (merge_prev) {
                    const prev_msg = try chatcmpl_messages.back();
                    if (!existsAndIsArray(prev_msg, "content")) {
                        try prev_msg.set(a, "content", json.Value.emptyArray());
                    }
                    const prev_content = try prev_msg.index(a, "content");
                    try prev_content.insert(a, &chatcmpl_content);
                } else {
                    try item.erase(a, "status");
                    try item.erase(a, "type");
                    try item.set(a, "content", chatcmpl_content);
                    try chatcmpl_messages.pushBack(a, item);
                }
            } else if (existsAndIsString(&item, "arguments") and
                existsAndIsString(&item, "call_id") and
                existsAndIsString(&item, "name") and
                existsAndIsString(&item, "type") and
                try atIs(&item, "type", "function_call"))
            {
                // #responses_create-input-input_item_list-item-function_tool_call
                const tool_call = try object(a, .{
                    .{ "function", try object(a, .{
                        .{ "arguments", try (try item.at("arguments")).clone(a) },
                        .{ "name", try (try item.at("name")).clone(a) },
                    }) },
                    .{ "id", try (try item.at("call_id")).clone(a) },
                    .{ "type", "function" },
                });

                if (merge_prev) {
                    const prev_msg = try chatcmpl_messages.back();
                    if (!existsAndIsArray(prev_msg, "tool_calls")) {
                        try prev_msg.set(a, "tool_calls", json.Value.emptyArray());
                    }
                    try (try prev_msg.index(a, "tool_calls")).pushBack(a, tool_call);
                } else {
                    try chatcmpl_messages.pushBack(a, try object(a, .{
                        .{ "role", "assistant" },
                        .{ "tool_calls", try arrayOf(a, tool_call) },
                    }));
                }
            } else if (existsAndIsString(&item, "call_id") and
                (existsAndIsString(&item, "output") or existsAndIsArray(&item, "output")) and
                existsAndIsString(&item, "type") and
                try atIs(&item, "type", "function_call_output"))
            {
                // #responses_create-input-input_item_list-item-function_tool_call_output
                if ((try item.at("output")).isString()) {
                    try chatcmpl_messages.pushBack(a, try object(a, .{
                        .{ "content", try (try item.at("output")).clone(a) },
                        .{ "role", "tool" },
                        .{ "tool_call_id", try (try item.at("call_id")).clone(a) },
                    }));
                } else {
                    const chatcmpl_outputs = try (try item.at("output")).clone(a);
                    for (chatcmpl_outputs.array.items) |*chatcmpl_output| {
                        if (!chatcmpl_output.contains("type")) {
                            return common_chat.exception.throw(error.InvalidArgument, "Output of tool call missing 'type' field", .{});
                        }
                        const is_text = try atIs(chatcmpl_output, "type", "input_text");
                        const is_image = try atIs(chatcmpl_output, "type", "input_image");
                        if (!is_text and !is_image) {
                            return common_chat.exception.throw(error.InvalidArgument, "Output of tool call should be 'Input text' or 'Input image'", .{});
                        }
                        if (is_text) {
                            try chatcmpl_output.set(a, "type", try json.Value.initString(a, "text"));
                        } else {
                            if (!chatcmpl_output.contains("image_url")) {
                                return common_chat.exception.throw(error.InvalidArgument, "'image_url' is required", .{});
                            }
                            try chatcmpl_output.set(a, "type", try json.Value.initString(a, "image_url"));
                            const url = try object(a, .{.{ "url", try (try chatcmpl_output.at("image_url")).clone(a) }});
                            try chatcmpl_output.set(a, "image_url", url);
                        }
                    }
                    try chatcmpl_messages.pushBack(a, try object(a, .{
                        .{ "content", chatcmpl_outputs },
                        .{ "role", "tool" },
                        .{ "tool_call_id", try (try item.at("call_id")).clone(a) },
                    }));
                }
            } else if (existsAndIsArray(&item, "summary") and
                existsAndIsString(&item, "type") and
                try atIs(&item, "type", "reasoning"))
            {
                // #responses_create-input-input_item_list-item-reasoning

                if (!existsAndIsArray(&item, "content")) {
                    return common_chat.exception.throw(error.InvalidArgument, "item['content'] is not an array", .{});
                }
                const content = try item.at("content");
                if (content.empty()) {
                    return common_chat.exception.throw(error.InvalidArgument, "item['content'] is empty", .{});
                }
                if (!existsAndIsString(&content.array.items[0], "text")) {
                    return common_chat.exception.throw(error.InvalidArgument, "item['content']['text'] is not a string", .{});
                }
                const text = try (try content.array.items[0].at("text")).clone(a);

                if (merge_prev) {
                    const prev_msg = try chatcmpl_messages.back();
                    try prev_msg.set(a, "reasoning_content", text);
                } else {
                    try chatcmpl_messages.pushBack(a, try object(a, .{
                        .{ "role", "assistant" },
                        .{ "content", json.Value.emptyArray() },
                        .{ "reasoning_content", text },
                    }));
                }
            } else {
                return common_chat.exception.throw(error.InvalidArgument, "Cannot determine type of 'item'", .{});
            }
        }
    } else {
        return common_chat.exception.throw(error.InvalidArgument, "'input' must be a string or array of objects", .{});
    }

    try chatcmpl_body.set(a, "messages", chatcmpl_messages);

    if (response_body.contains("tools")) {
        const chatcmpl_tools = try responsesTools(a, try response_body.at("tools"));
        try chatcmpl_body.erase(a, "tools");
        if (!chatcmpl_tools.empty()) {
            try chatcmpl_body.set(a, "tools", chatcmpl_tools);
        }
    }

    if (response_body.contains("max_output_tokens")) {
        try chatcmpl_body.erase(a, "max_output_tokens");
        try chatcmpl_body.set(a, "max_tokens", try (try response_body.at("max_output_tokens")).clone(a));
    }

    if (response_body.contains("reasoning")) {
        // Only "effort" is handled so far
        const reasoning = try response_body.at("reasoning");
        if (reasoning.contains("effort")) {
            try chatcmpl_body.set(a, "reasoning_effort", try (try reasoning.at("effort")).clone(a));
        }
        try chatcmpl_body.erase(a, "reasoning");
    }

    if (response_body.contains("text")) {
        const text = try response_body.at("text");
        if (!text.isObject()) {
            return common_chat.exception.throw(error.InvalidArgument, "'text' must be an object", .{});
        }

        try chatcmpl_body.erase(a, "text");
        if (text.contains("format")) {
            const format = try text.at("format");
            if (!format.isObject()) {
                return common_chat.exception.throw(error.InvalidArgument, "'text.format' must be an object", .{});
            }

            const ty = mod.common.jsonValue(format, "type", []const u8, "");
            if (std.mem.eql(u8, ty, "json_schema")) {
                var rf = json.Value.emptyObject();
                try rf.set(a, "type", try json.Value.initString(a, "json_schema"));
                try rf.set(a, "json_schema", try format.clone(a));
                try chatcmpl_body.set(a, "response_format", rf);
                try (try (try chatcmpl_body.index(a, "response_format")).index(a, "json_schema")).erase(a, "type");
            } else if (std.mem.eql(u8, ty, "json_object") or std.mem.eql(u8, ty, "text")) {
                try chatcmpl_body.set(a, "response_format", try format.clone(a));
            } else if (ty.len != 0) {
                return common_chat.exception.throw(error.InvalidArgument, "'text.format.type' must be one of 'text', 'json_object', or 'json_schema'", .{});
            }
        }
    }

    // llama.cpp params (e.g. seed, min_p) go under extra_params and are merged into the converted body
    if (response_body.contains("extra_params")) {
        try chatcmpl_body.erase(a, "extra_params");
        const extra_params = try response_body.at("extra_params");
        if (extra_params.isObject()) {
            var it = @constCast(extra_params).items();
            while (it.next()) |item| {
                try chatcmpl_body.set(a, item.key, try item.value.clone(a));
            }
        }
    }

    return chatcmpl_body;
}

// -----------------------------------------------------------------------------
// the Anthropic Messages API

/// Ports `normalize_anthropic_billing_header` (server-chat.cpp:369
/// @b1b770d90): the five-character `cch=` stamp of an
/// `x-anthropic-billing-header:` system prompt overwritten with `f`s, so it
/// does not break prefix caching. Any other prompt is left alone.
///
/// Parameters:
/// - `system_text`: edited in place.
pub fn normalizeAnthropicBillingHeader(system_text: []u8) void {
    const prefix = "x-anthropic-billing-header:";
    if (!std.mem.startsWith(u8, system_text, prefix)) {
        return;
    }

    const header_prefix_length = prefix.len;
    const cch_length = 5;
    const index_cch = std.mem.indexOfPos(u8, system_text, header_prefix_length, "cch=") orelse return;

    const index_replace = index_cch + 4;
    if (index_replace + cch_length < system_text.len and system_text[index_replace + cch_length] == ';') {
        for (0..cch_length) |i| {
            system_text[index_replace + i] = 'f';
        }
    } else {
        const z = mod.common.dupZ(system_text);
        defer mod.common.freeZ(z);
        tools_common.log.err("anthropic string not as expected: %s", .{z});
    }
}

/// A `source` block's image as a chat `image_url` part: a data URI for
/// `base64`, the URL for `url`, nothing for any other type.
fn imageUrlOf(a: std.mem.Allocator, source: *const json.Value) common_chat.exception.Error!?json.Value {
    const source_type = mod.common.jsonValue(source, "type", []const u8, "");
    if (std.mem.eql(u8, source_type, "base64")) {
        const media_type = mod.common.jsonValue(source, "media_type", []const u8, "image/jpeg");
        const data = mod.common.jsonValue(source, "data", []const u8, "");
        const url = try std.mem.concat(a, u8, &.{ "data:", media_type, ";base64,", data });
        return try object(a, .{
            .{ "type", "image_url" },
            .{ "image_url", try object(a, .{.{ "url", url }}) },
        });
    } else if (std.mem.eql(u8, source_type, "url")) {
        const url = mod.common.jsonValue(source, "url", []const u8, "");
        return try object(a, .{
            .{ "type", "image_url" },
            .{ "image_url", try object(a, .{.{ "url", url }}) },
        });
    }
    return null;
}

/// Ports `server_chat_convert_anthropic_to_oai` (server-chat.cpp:391
/// @b1b770d90): an Anthropic Messages request as a `/chat/completions`
/// one -- the system prompt a message, content blocks as parts, thinking
/// blocks as reasoning, `tool_use` blocks as tool calls and `tool_result`
/// blocks as tool messages after their turn; tools, `tool_choice`,
/// `stop_sequences`, `thinking` and `metadata.user_id` renamed.
///
/// Parameters:
/// - `a`: an arena; the result is its.
///
/// Return: the body; `error.RuntimeError` without `messages`.
pub fn anthropicToOai(a: std.mem.Allocator, body: *const json.Value) common_chat.exception.Error!json.Value {
    var oai_body: json.Value = .null;

    // Convert system prompt
    var oai_messages = json.Value.emptyArray();
    const system_param = try mod.common.jsonValueJson(a, body, "system", .null);
    if (!system_param.isNull()) {
        var system_content: std.ArrayList(u8) = .empty;

        if (system_param.isString()) {
            try system_content.appendSlice(a, system_param.string);
            normalizeAnthropicBillingHeader(system_content.items);
        } else if (system_param.isArray()) {
            for (system_param.array.items) |*block| {
                if (std.mem.eql(u8, mod.common.jsonValue(block, "type", []const u8, ""), "text")) {
                    const system_text = try a.dupe(u8, mod.common.jsonValue(block, "text", []const u8, ""));
                    normalizeAnthropicBillingHeader(system_text);
                    try system_content.appendSlice(a, system_text);
                }
            }
        }

        try oai_messages.pushBack(a, try object(a, .{
            .{ "role", "system" },
            .{ "content", system_content.items },
        }));
    }

    // Convert messages
    if (!body.contains("messages")) {
        return common_chat.exception.throw(error.RuntimeError, "'messages' is required", .{});
    }
    const messages = try body.at("messages");
    if (messages.isArray()) {
        for (messages.array.items) |*msg| {
            const role = mod.common.jsonValue(msg, "role", []const u8, "");

            if (!msg.contains("content")) {
                if (std.mem.eql(u8, role, "assistant")) {
                    continue;
                }
                try oai_messages.pushBack(a, try msg.clone(a));
                continue;
            }

            const content = try msg.at("content");

            if (content.isString()) {
                try oai_messages.pushBack(a, try msg.clone(a));
                continue;
            }

            if (!content.isArray()) {
                try oai_messages.pushBack(a, try msg.clone(a));
                continue;
            }

            var tool_calls = json.Value.emptyArray();
            var converted_content = json.Value.emptyArray();
            var tool_results = json.Value.emptyArray();
            var reasoning_content: std.ArrayList(u8) = .empty;
            var has_tool_calls = false;

            for (content.array.items) |*block| {
                const ty = mod.common.jsonValue(block, "type", []const u8, "");

                if (std.mem.eql(u8, ty, "text")) {
                    try converted_content.pushBack(a, try block.clone(a));
                } else if (std.mem.eql(u8, ty, "thinking")) {
                    try reasoning_content.appendSlice(a, mod.common.jsonValue(block, "thinking", []const u8, ""));
                } else if (std.mem.eql(u8, ty, "image")) {
                    const source = try mod.common.jsonValueJson(a, block, "source", json.Value.emptyObject());
                    if (try imageUrlOf(a, &source)) |part| {
                        try converted_content.pushBack(a, part);
                    }
                } else if (std.mem.eql(u8, ty, "tool_use")) {
                    const input = try mod.common.jsonValueJson(a, block, "input", json.Value.emptyObject());
                    try tool_calls.pushBack(a, try object(a, .{
                        .{ "id", mod.common.jsonValue(block, "id", []const u8, "") },
                        .{ "type", "function" },
                        .{ "function", try object(a, .{
                            .{ "name", mod.common.jsonValue(block, "name", []const u8, "") },
                            .{ "arguments", try json.dump(a, &input, -1) },
                        }) },
                    }));
                    has_tool_calls = true;
                } else if (std.mem.eql(u8, ty, "tool_result")) {
                    const tool_use_id = mod.common.jsonValue(block, "tool_use_id", []const u8, "");

                    const result_content = try mod.common.jsonValueJson(a, block, "content", .null);
                    if (result_content.isString()) {
                        try tool_results.pushBack(a, try object(a, .{
                            .{ "role", "tool" },
                            .{ "tool_call_id", tool_use_id },
                            .{ "content", result_content.string },
                        }));
                    } else if (result_content.isArray()) {
                        // Single-pass: build both text and content_parts, decide format at the end
                        var result_text: std.ArrayList(u8) = .empty;
                        var content_parts = json.Value.emptyArray();
                        var keep_parts = false; // images or tool references

                        for (result_content.array.items) |*cb| {
                            const c_type = mod.common.jsonValue(cb, "type", []const u8, "");
                            if (std.mem.eql(u8, c_type, "text")) {
                                const text = mod.common.jsonValue(cb, "text", []const u8, "");
                                try result_text.appendSlice(a, text);
                                try content_parts.pushBack(a, try object(a, .{
                                    .{ "type", "text" },
                                    .{ "text", text },
                                }));
                            } else if (std.mem.eql(u8, c_type, "image")) {
                                keep_parts = true;
                                const source = try mod.common.jsonValueJson(a, cb, "source", json.Value.emptyObject());
                                if (try imageUrlOf(a, &source)) |part| {
                                    try content_parts.pushBack(a, part);
                                }
                            } else if (std.mem.eql(u8, c_type, "tool_reference")) {
                                // tool search result naming a deferred tool
                                keep_parts = true;
                                try content_parts.pushBack(a, try object(a, .{
                                    .{ "type", "tool_reference" },
                                    .{ "name", mod.common.jsonValue(cb, "tool_name", []const u8, "") },
                                }));
                            }
                        }

                        if (!keep_parts) {
                            // Text-only: collapse to a plain string for maximum compatibility
                            try tool_results.pushBack(a, try object(a, .{
                                .{ "role", "tool" },
                                .{ "tool_call_id", tool_use_id },
                                .{ "content", result_text.items },
                            }));
                        } else {
                            // images or tool references: keep array content parts
                            try tool_results.pushBack(a, try object(a, .{
                                .{ "role", "tool" },
                                .{ "tool_call_id", tool_use_id },
                                .{ "content", content_parts },
                            }));
                        }
                    } else {
                        try tool_results.pushBack(a, try object(a, .{
                            .{ "role", "tool" },
                            .{ "tool_call_id", tool_use_id },
                            .{ "content", "" },
                        }));
                    }
                }
            }

            if (!converted_content.empty() or has_tool_calls or reasoning_content.items.len != 0) {
                var new_msg = try object(a, .{.{ "role", role }});
                if (!converted_content.empty()) {
                    try new_msg.set(a, "content", converted_content);
                } else if (has_tool_calls or reasoning_content.items.len != 0) {
                    try new_msg.set(a, "content", try json.Value.initString(a, ""));
                }
                if (!tool_calls.empty()) {
                    try new_msg.set(a, "tool_calls", tool_calls);
                }
                if (reasoning_content.items.len != 0) {
                    try new_msg.set(a, "reasoning_content", try json.Value.initString(a, reasoning_content.items));
                }
                try oai_messages.pushBack(a, new_msg);
            }

            for (tool_results.array.items) |tool_msg| {
                try oai_messages.pushBack(a, tool_msg);
            }
        }
    }

    try oai_body.set(a, "messages", oai_messages);

    // Convert tools
    if (body.contains("tools")) {
        const tools_in = try body.at("tools");
        if (tools_in.isArray()) {
            var oai_tools = json.Value.emptyArray();
            for (tools_in.array.items) |*tool| {
                const parameters = if (tool.contains("input_schema")) try (try tool.at("input_schema")).clone(a) else json.Value.emptyObject();
                try oai_tools.pushBack(a, try object(a, .{
                    .{ "type", "function" },
                    .{ "function", try object(a, .{
                        .{ "name", mod.common.jsonValue(tool, "name", []const u8, "") },
                        .{ "description", mod.common.jsonValue(tool, "description", []const u8, "") },
                        .{ "parameters", parameters },
                    }) },
                }));
                if (tool.contains("defer_loading")) {
                    try (try oai_tools.back()).set(a, "defer_loading", try (try tool.at("defer_loading")).clone(a));
                }
            }
            try oai_body.set(a, "tools", oai_tools);
        }
    }

    // Convert tool_choice
    if (body.contains("tool_choice")) {
        const tc = try body.at("tool_choice");
        if (tc.isObject()) {
            const ty = mod.common.jsonValue(tc, "type", []const u8, "");
            if (std.mem.eql(u8, ty, "auto")) {
                try oai_body.set(a, "tool_choice", try json.Value.initString(a, "auto"));
            } else if (std.mem.eql(u8, ty, "any") or std.mem.eql(u8, ty, "tool")) {
                try oai_body.set(a, "tool_choice", try json.Value.initString(a, "required"));
            }
        }
    }

    // Convert stop_sequences to stop
    if (body.contains("stop_sequences")) {
        try oai_body.set(a, "stop", try (try body.at("stop_sequences")).clone(a));
    }

    // Handle max_tokens (required in Anthropic, but we're permissive)
    if (body.contains("max_tokens")) {
        try oai_body.set(a, "max_tokens", try (try body.at("max_tokens")).clone(a));
    } else {
        try oai_body.set(a, "max_tokens", .{ .number_integer = 4096 });
    }

    // Pass through common params
    for ([_][]const u8{ "temperature", "top_p", "top_k", "stream", "chat_template_kwargs", "id_slot" }) |key| {
        if (body.contains(key)) {
            try oai_body.set(a, key, try (try body.at(key)).clone(a));
        }
    }

    // Handle Anthropic-specific thinking param
    if (body.contains("thinking")) {
        const thinking = try mod.common.jsonValueJson(a, body, "thinking", json.Value.emptyObject());
        const thinking_type = mod.common.jsonValue(&thinking, "type", []const u8, "");
        if (std.mem.eql(u8, thinking_type, "enabled")) {
            const budget_tokens = mod.common.jsonValue(&thinking, "budget_tokens", c_int, 10000);
            try oai_body.set(a, "thinking_budget_tokens", .{ .number_integer = budget_tokens });
        }
    }

    // Handle Anthropic-specific metadata param
    if (body.contains("metadata")) {
        const metadata = try mod.common.jsonValueJson(a, body, "metadata", json.Value.emptyObject());
        const user_id = mod.common.jsonValue(&metadata, "user_id", []const u8, "");
        if (user_id.len != 0) {
            try oai_body.set(a, "__metadata_user_id", try json.Value.initString(a, user_id));
        }
    }

    // llama.cpp params (e.g. seed, min_p) go under extra_params and are merged into the converted body
    if (body.contains("extra_params") and (try body.at("extra_params")).isObject()) {
        for ((try body.at("extra_params")).object.members.items) |*member| {
            try oai_body.set(a, member.key, try member.value.clone(a));
        }
    }

    return oai_body;
}

// -----------------------------------------------------------------------------
// the transcriptions API

/// Ports libc++'s `std::stoul` (`as_integer` over `strtoul`, base 10) on
/// `c_str()`: `invalid_argument` "stoul: no conversion" when nothing
/// parses, `out_of_range` "stoul: out of range" on `ERANGE` (thrown here
/// as `error.RuntimeError`, which the server answers as it answers any
/// exception but `invalid_argument`). A minus sign negates modulo 2^64.
pub fn stoul(s: []const u8) common_chat.exception.Error!c_ulong {
    const p = stdlib.num.strtoul(std.mem.sliceTo(s, 0), 10);
    if (p.range) return common_chat.exception.throw(error.RuntimeError, "stoul: out of range", .{});
    if (p.end == 0) return common_chat.exception.throw(error.InvalidArgument, "stoul: no conversion", .{});
    return p.value;
}

/// Ports libc++'s `std::stoll` (`as_integer` over `strtoll`, base 10) on
/// `c_str()`, as `stoul` does.
pub fn stoll(s: []const u8) common_chat.exception.Error!c_longlong {
    const p = stdlib.num.strtoll(std.mem.sliceTo(s, 0), 10);
    if (p.range) return common_chat.exception.throw(error.RuntimeError, "stoll: out of range", .{});
    if (p.end == 0) return common_chat.exception.throw(error.InvalidArgument, "stoll: no conversion", .{});
    return p.value;
}

/// Ports libc++'s `std::stof` (`as_float` over `strtof`) on `c_str()`, as
/// `stoul` does.
pub fn stof(s: []const u8) common_chat.exception.Error!f32 {
    const p = stdlib.num.strtof(std.mem.sliceTo(s, 0));
    if (p.range) return common_chat.exception.throw(error.RuntimeError, "stof: out of range", .{});
    if (p.end == 0) return common_chat.exception.throw(error.InvalidArgument, "stof: no conversion", .{});
    return p.value;
}

/// Ports `convert_transcriptions_to_chatcmpl` (server-chat.cpp:711
/// @b1b770d90): a transcription request -- its form fields as JSON, its
/// uploaded `file` -- as a `/chat/completions` body: the template's ASR
/// preset as the system and user messages, the prompt ending in the media
/// marker, and the form's string fields typed.
///
/// Parameters:
/// - `a`: an arena; the result and the file are its.
/// - `tmpls`: the server's templates, for the ASR preset.
/// - `in_file`: `in_files.find("file")`: the request's upload of that
///   name, or null.
/// - `out_files`: cleared, then holds the `file` upload.
///
/// Return: the body; `error.InvalidArgument` without a file or for a
/// response format other than `json`.
pub fn transcriptionsToChatcmpl(
    a: std.mem.Allocator,
    inp_body: *const json.Value,
    tmpls: *const common_chat.templates.Templates,
    in_file: ?[]const u8,
    out_files: *std.ArrayList([]u8),
) common_chat.exception.Error!json.Value {
    // TODO @ngxson : this function may need to be improved in the future
    // handle input files
    out_files.clearRetainingCapacity();
    if (in_file) |f| {
        try out_files.append(a, try a.dupe(u8, f));
    } else {
        return common_chat.exception.throw(error.InvalidArgument, "No input file found for transcription", .{});
    }

    // handle input data
    var prompt: std.ArrayList(u8) = .empty;
    try prompt.appendSlice(a, mod.common.jsonValue(inp_body, "prompt", []const u8, ""));
    const language = mod.common.jsonValue(inp_body, "language", []const u8, "");
    const response_format = mod.common.jsonValue(inp_body, "response_format", []const u8, "json");
    if (!std.mem.eql(u8, response_format, "json")) {
        return common_chat.exception.throw(error.InvalidArgument, "Only 'json' response_format is supported for transcription", .{});
    }
    const preset = try common_chat.templates.getAsrPrompt(a, tmpls, language);
    if (prompt.items.len == 0) {
        try prompt.appendSlice(a, preset.user);
    }
    if (language.len != 0 and !preset.has_language) {
        // string_format's %s stops at a NUL
        const lang = language[0 .. std.mem.indexOfScalar(u8, language, 0) orelse language.len];
        try prompt.print(a, " (language: {s})", .{lang});
    }
    try prompt.appendSlice(a, mod.common.getMediaMarker());

    var messages = json.Value.emptyArray();
    if (preset.system.len != 0) {
        try messages.pushBack(a, try object(a, .{ .{ "role", "system" }, .{ "content", preset.system } }));
    }
    try messages.pushBack(a, try object(a, .{ .{ "role", "user" }, .{ "content", prompt.items } }));

    var chatcmpl_body = try inp_body.clone(a); // copy all fields
    try chatcmpl_body.set(a, "messages", messages);

    // because input from form-data, everything is string, we need to correct the types here
    const stream = mod.common.jsonValue(inp_body, "stream", []const u8, "false");
    try chatcmpl_body.set(a, "stream", .{ .boolean = std.mem.eql(u8, stream, "true") });

    if (inp_body.contains("max_tokens")) {
        const inp = try (try inp_body.at("max_tokens")).get([]const u8);
        try chatcmpl_body.set(a, "max_tokens", .{ .number_unsigned = try stoul(inp) });
    }

    if (inp_body.contains("temperature")) {
        const inp = try (try inp_body.at("temperature")).get([]const u8);
        try chatcmpl_body.set(a, "temperature", try json.Value.from(a, try stof(inp)));
    }

    return chatcmpl_body;
}

// -----------------------------------------------------------------------------
// the speech API

/// Ports `convert_speech_to_tts` (server-chat.cpp:768 @b1b770d90): an OpenAI
/// speech request as a `/tts` one -- a custom voice `{"id": ...}` as its
/// name, `response_format` `wav` or `pcm` (16-bit, `pcm16` in `/tts`),
/// `stream_format` `audio` as streaming; `speed` other than 1 is refused
/// and `instructions` ignored with a warning.
///
/// Parameters:
/// - `a`: an arena; the result is its.
///
/// Return: the body; `error.InvalidArgument` for what it cannot convert.
pub fn speechToTts(a: std.mem.Allocator, inp_body: *const json.Value) common_chat.exception.Error!json.Value {
    var tts_body = try inp_body.clone(a); // copy all fields

    // a custom voice is given as {"id": "..."}
    if (inp_body.contains("voice") and (try inp_body.at("voice")).isObject()) {
        try tts_body.set(a, "voice", try json.Value.initString(a, mod.common.jsonValue(try inp_body.at("voice"), "id", []const u8, "")));
    }

    // the default of the OpenAI API is mp3, we have no encoder for it
    const response_format = mod.common.jsonValue(inp_body, "response_format", []const u8, "wav");
    if (std.mem.eql(u8, response_format, "pcm")) {
        try tts_body.set(a, "response_format", try json.Value.initString(a, "pcm16")); // "pcm" is float32 in /tts
    } else if (std.mem.eql(u8, response_format, "wav")) {
        try tts_body.set(a, "response_format", try json.Value.initString(a, "wav"));
    } else {
        return common_chat.exception.throw(error.InvalidArgument, "Only 'wav' and 'pcm' response_format are supported for speech", .{});
    }

    const stream_format = mod.common.jsonValue(inp_body, "stream_format", []const u8, "");
    if (std.mem.eql(u8, stream_format, "audio")) {
        try tts_body.set(a, "stream", .{ .boolean = true });
    } else if (stream_format.len != 0) {
        return common_chat.exception.throw(error.InvalidArgument, "Only 'audio' stream_format is supported for speech", .{});
    }

    if (mod.common.jsonValue(inp_body, "speed", f64, 1.0) != 1.0) {
        return common_chat.exception.throw(error.InvalidArgument, "'speed' is not supported for speech", .{});
    }

    if (mod.common.jsonValue(inp_body, "instructions", []const u8, "").len != 0) {
        mod.common.srv(.wrn, "convert_speech_to_tts", "%s", .{@as([*:0]const u8, "'instructions' is not supported for speech and is ignored\n")});
    }

    return tts_body;
}

// -----------------------------------------------------------------------------
// the "asr" chat template

/// Ports `server_chat_get_asr_prompt` (server-chat.cpp:804 @b1b770d90):
/// prompt of the "asr" chat template, live ASR models (e.g. LFM3-Audio live
/// ASR) transcribe with it; empty if the model has no such template,
/// `error.RuntimeError` on a template error.
///
/// Return: owned by `gpa`.
pub fn getAsrPrompt(gpa: std.mem.Allocator, model: ?*const llama.impl.llama_c.llama_model) common_chat.exception.Error![]llama.impl.llama_c.llama_token {
    const src = llama.api.model.llama_model_chat_template(model, "asr");
    if (src == null) {
        return gpa.alloc(llama.impl.llama_c.llama_token, 0);
    }
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const vocab = llama.api.model.llama_model_get_vocab(model);
    const bos = llama.api.vocab.llama_vocab_bos(vocab);
    const bos_text: []const u8 = if (bos == mod.common.token_null) "" else try tools_common.common.tokenToPiece(a, vocab, bos, true);
    var tmpl = try common_chat.template.Template.init(a, std.mem.span(src), bos_text, "");
    defer tmpl.deinit();

    var inp = json.Value.emptyObject();
    try inp.set(a, "bos_token", try json.Value.initString(a, tmpl.bosToken()));
    try inp.set(a, "messages", json.Value.emptyArray());
    try inp.set(a, "add_generation_prompt", .{ .boolean = true });

    var r: common_chat.template.Render = undefined;
    r.init(a, false);
    defer r.deinit();
    try r.bindJson(&inp, false);
    const prompt = try r.run(&tmpl, a);
    return gpa.dupe(llama.impl.llama_c.llama_token, try mod.common.tokenize(a, vocab, prompt, false, true));
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a string input is one user message" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var body = try json.parse(a, "{\"input\":\"hi\",\"instructions\":\"be brief\",\"max_output_tokens\":5}");
    const out = try responsesToChatcmpl(a, &body);
    try std.testing.expectEqualStrings(
        "{\"messages\":[{\"role\":\"system\",\"content\":\"be brief\"},{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}",
        try json.dump(a, &out, -1),
    );
}

test "the billing stamp is overwritten" {
    var s = "x-anthropic-billing-header: cc_version=2; cch=a5145;You are".*;
    normalizeAnthropicBillingHeader(&s);
    try std.testing.expectEqualStrings("x-anthropic-billing-header: cc_version=2; cch=fffff;You are", &s);
}
