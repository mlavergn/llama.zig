//! The OpenAI-compatible request conversions -- `/completions` and
//! `/chat/completions` bodies into llama.cpp's native parameters -- the
//! media a chat message carries, and the embeddings and rerank responses.
//!
//! # Provenance
//!
//! Ports `oaicompat_completion_params_parse`, `handle_media`,
//! `server_chat_params`, `oaicompat_chat_params_parse`,
//! `format_embeddings_response_oaicompat` and `format_response_rerank` from
//! `llama.cpp/tools/server/server-common.cpp` and
//! `llama.cpp/tools/server/server-common.h`; `fs_validate_filename` from
//! `llama.cpp/common/common.cpp`; `common_chat_templates_inputs` and
//! `common_chat_templates_apply` from `llama.cpp/common/chat.h` and
//! `llama.cpp/common/chat.cpp`, over `cli/common_chat/templates.zig`'s
//! port of both arms; and `base64::encode` from
//! `llama.cpp/common/base64.hpp` (llama.cpp v1.0.0, `b1b770d90`). Each
//! declaration names what it ports and the line it began at.
//!
//! # Remote media
//!
//! `handle_media`'s download of an `http(s)` URL goes through
//! `download.zig`'s `common_remote_get_content`: plain
//! HTTP only, as the reference has no SSL.

const std = @import("std");
const mod = @import("module.zig");
const tc = @import("tools_common");
const stdlib = @import("stdlib");
const json = @import("json");
const chat = @import("common_chat");

const Allocator = std.mem.Allocator;

/// `DIRECTORY_SEPARATOR` (common/common.h:31 @b1b770d90).
const directory_separator: u8 = if (@import("builtin").os.tag == .windows) '\\' else '/';

// -----------------------------------------------------------------------------
// /completions

/// Ports `oaicompat_completion_params_parse` (server-common.cpp:1177 @b1b770d90):
/// an OpenAI `/completions` body as llama.cpp's own parameters.
///
/// Return: owned by `a`.
pub fn oaicompatCompletionParamsParse(a: Allocator, body: *const json.Value) mod.common.Error!json.Value {
    var llama_params: json.Value = .null;
    errdefer llama_params.deinit(a);

    if (!body.contains("prompt")) {
        return mod.common.throw(error.RuntimeError, "\"prompt\" is required", .{});
    }

    try handleStop(a, &llama_params, body);

    // Handle "echo" field
    if (mod.common.jsonValue(body, "echo", bool, false)) {
        return mod.common.throw(error.RuntimeError, "Only no echo is supported", .{});
    }

    // Params supported by OAI but unsupported by llama.cpp
    const unsupported_params = [_][]const u8{ "best_of", "suffix" };
    for (unsupported_params) |param| {
        if (body.contains(param)) {
            return mod.common.throw(error.RuntimeError, "Unsupported param: {s}", .{param});
        }
    }

    try copyRemaining(a, &llama_params, body);

    return llama_params;
}

/// The "stop" field both conversions share: a string becomes a one-element
/// array.
fn handleStop(a: Allocator, llama_params: *json.Value, body: *const json.Value) mod.common.Error!void {
    // Handle "stop" field
    if (body.contains("stop") and (try body.at("stop")).isString()) {
        var arr = json.Value.emptyArray();
        errdefer arr.deinit(a);
        try arr.pushBack(a, try json.Value.initString(a, try (try body.at("stop")).get([]const u8)));
        try llama_params.set(a, "stop", arr);
    } else {
        try llama_params.set(a, "stop", try mod.common.jsonValueJson(a, body, "stop", json.Value.emptyArray()));
    }
}

/// Copies the remaining properties: every member not already set, and
/// `n_predict` whatever was set before ("max_tokens" may have).
fn copyRemaining(a: Allocator, llama_params: *json.Value, body: *const json.Value) mod.common.Error!void {
    var it = @constCast(body).items();
    while (it.next()) |item| {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key) or std.mem.eql(u8, item.key, "n_predict")) {
            try llama_params.set(a, item.key, try item.value.clone(a));
        }
    }
}

// -----------------------------------------------------------------------------
// media

/// Ports `fs_validate_filename` (common/common.cpp:824 @b1b770d90): no control
/// characters, no lookalikes of path syntax, no `..`, no leading or
/// trailing space and no trailing dot.
pub fn fsValidateFilename(filename: []const u8, allow_subdirs: bool) bool {
    if (filename.len == 0) {
        // Empty filename invalid
        return false;
    }
    if (filename.len > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    var offset: usize = 0;
    while (offset < filename.len) {
        const result = json.trie.parseUtf8Codepoint(filename, offset);

        if (result.status != .success) {
            return false;
        }
        const ch = result.codepoint;

        if ((result.bytes_consumed == 2 and ch < 0x80) or
            (result.bytes_consumed == 3 and ch < 0x800) or
            (result.bytes_consumed == 4 and ch < 0x10000))
        {
            return false;
        }

        // Check for forbidden codepoints:
        // - Control characters
        // - Unicode equivalents of illegal characters
        // - UTF-16 surrogate pairs
        // - UTF-8 replacement character
        // - Byte order mark (BOM)
        // - Illegal characters: / \ : * ? " < > |
        if (ch <= 0x1F // Control characters (C0)
        or ch == 0x7F // Control characters (DEL)
        or (ch >= 0x80 and ch <= 0x9F) // Control characters (C1)
        or ch == 0xFF0E // Fullwidth Full Stop (period equivalent)
        or ch == 0x2215 // Division Slash (forward slash equivalent)
        or ch == 0x2216 // Set Minus (backslash equivalent)
        or (ch >= 0xD800 and ch <= 0xDFFF) // UTF-16 surrogate pairs
        or ch > 0x10FFFF // Max Unicode limit
        or ch == 0xFFFD // Replacement Character (UTF-8)
        or ch == 0xFEFF // Byte Order Mark (BOM)
        or ch == ':' or ch == '*' // Illegal characters
        or ch == '?' or ch == '"' or ch == '<' or ch == '>' or ch == '|') {
            return false;
        }
        if (!allow_subdirs and (ch == '/' or ch == '\\')) {
            // Subdirectories not allowed, reject path separators
            return false;
        }
        offset += result.bytes_consumed;
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename[0] == ' ' or filename[filename.len - 1] == ' ' or filename[filename.len - 1] == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (std.mem.indexOf(u8, filename, "..") != null) {
        return false;
    }

    // Reject "."
    if (std.mem.eql(u8, filename, ".")) {
        return false;
    }

    return true;
}

const AF_INET: c_int = 2;
const AF_INET6: c_int = if (@import("builtin").os.tag.isDarwin()) 30 else 10;

/// Ports `ipv4_is_public` (server-common.cpp:1220 @b1b770d90).
fn ipv4IsPublic(b: []const u8) bool {
    return !(b[0] == 0 or b[0] == 10 or b[0] == 127 or b[0] >= 224 // this network, private, loopback, multicast, reserved
    or (b[0] == 100 and (b[1] & 0xc0) == 64) // 100.64.0.0/10 CGNAT
    or (b[0] == 169 and b[1] == 254) // link-local, cloud metadata
    or (b[0] == 172 and (b[1] & 0xf0) == 16) // 172.16.0.0/12
    or (b[0] == 192 and b[1] == 168) // 192.168.0.0/16
    or (b[0] == 192 and b[1] == 0 and b[2] == 0) // 192.0.0.0/24
    or (b[0] == 198 and (b[1] & 0xfe) == 18)); // 198.18.0.0/15
}

/// Ports `server_addr_is_public` (server-common.cpp:1230 @b1b770d90): true
/// if the address is publicly routable.
pub fn serverAddrIsPublic(addr: []const u8) bool {
    var zbuf: [256]u8 = undefined;
    // c_str(): inet_pton reads up to the first NUL
    const n = @min(std.mem.indexOfScalar(u8, addr, 0) orelse addr.len, zbuf.len - 1);
    @memcpy(zbuf[0..n], addr[0..n]);
    zbuf[n] = 0;
    const z: [*:0]const u8 = zbuf[0..n :0];
    var b: [16]u8 = undefined;
    if (stdlib.libc.inet_pton(AF_INET, z, &b) == 1) {
        return ipv4IsPublic(&b);
    }
    if (stdlib.libc.inet_pton(AF_INET6, z, &b) != 1) {
        return false; // e.g. scoped address "fe80::1%eth0"
    }
    const v4_mapped = [12]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff };
    const nat64 = [12]u8{ 0, 0x64, 0xff, 0x9b, 0, 0, 0, 0, 0, 0, 0, 0 };
    const zero: [12]u8 = @splat(0);
    if (std.mem.eql(u8, b[0..12], &v4_mapped) or std.mem.eql(u8, b[0..12], &nat64)) {
        return ipv4IsPublic(b[12..]);
    }
    return !std.mem.eql(u8, b[0..12], &zero) // ::/96: unspecified, loopback, IPv4-compatible
    and (b[0] & 0xfe) != 0xfc // fc00::/7 unique local
    and !(b[0] == 0xfe and (b[1] & 0x80) == 0x80) // fe80::/10 link-local, fec0::/10 site-local
    and b[0] != 0xff; // multicast
}

/// Ports `server_resolve_public_addr` (server-common.cpp:1250 @b1b770d90):
/// `host` resolved to an address to connect to; throws if it does not
/// resolve, or if any of its addresses is not public (unless
/// `allow_private`).
///
/// Return: the first address, owned by `a`.
pub fn serverResolvePublicAddr(a: Allocator, host: []const u8, allow_private: bool) mod.common.Error![]const u8 {
    var addrs: std.ArrayList([]const u8) = .empty;
    try mod.httplib.hostedAt(a, host, &addrs);
    if (addrs.items.len == 0) {
        return mod.common.throw(error.InvalidArgument, "cannot resolve host: {s}", .{host});
    }
    for (addrs.items) |addr| {
        if (!allow_private and !serverAddrIsPublic(addr)) {
            return mod.common.throw(error.InvalidArgument, "host is not a public address: {s}", .{host});
        }
    }
    return addrs.items[0];
}

/// Ports `resolve_host` (server-common.cpp:1280 @b1b770d90), the lambda
/// handle_media sets; `ctx` is the `allow_private_urls` it captured.
fn resolveHostForMedia(ctx: ?*const anyopaque, a: Allocator, host: []const u8) mod.common.Error![]const u8 {
    const allow_private: *const bool = @ptrCast(ctx.?);
    return serverResolvePublicAddr(a, host, allow_private.*);
}

/// Ports `handle_media` (server-common.cpp:1264 @b1b770d90): a media URL --
/// `http(s)://`, `file://` under `media_path`, a `data:` URI, or bare
/// base64 -- as bytes appended to `out_files`. An `http(s)` URL must
/// resolve to public addresses unless `allow_private_urls`.
///
/// Parameters:
/// - `out_files`: grows; each file owned by `a`.
pub fn handleMedia(a: Allocator, out_files: *std.ArrayList([]u8), url: []const u8, media_path: []const u8, allow_private_urls: bool) mod.common.Error!void {
    if (media_path.len != 0) {
        // should already be enforced by arg.cpp, but checking just in case
        tc.ggml.ggmlAssert(media_path[media_path.len - 1] == directory_separator, "server-common.cpp", 1271, "media_path.back() == DIRECTORY_SEPARATOR");
    }

    if (std.mem.startsWith(u8, url, "http")) {
        // download remote image
        const u = mod.common.dupZ(url);
        defer mod.common.freeZ(u);
        mod.common.srv(.inf, "handle_media", "downloading image from '%s'\n", .{u});
        // TODO @ngxson : maybe make these params configurable
        const params: mod.download.RemoteParams = .{
            .max_size = 1024 * 1024 * 10, // 10MB
            .timeout = 10, // seconds
            .resolve_host = .{ .ctx = &allow_private_urls, .call = resolveHostForMedia },
        };
        const res = try mod.download.remoteGetContent(a, url, &params);
        if (200 <= res.status and res.status < 300) {
            mod.common.srv(.inf, "handle_media", "downloaded %zu bytes\n", .{res.body.len});
            try out_files.append(a, res.body);
        } else {
            return mod.common.throw(error.RuntimeError, "Failed to download image", .{});
        }
    } else if (std.mem.startsWith(u8, url, "file://")) {
        if (media_path.len == 0) {
            return mod.common.throw(error.InvalidArgument, "file:// URLs are not allowed unless --media-path is specified", .{});
        }
        // load local image file
        const file_path = url[7..]; // remove "file://"
        if (!fsValidateFilename(file_path, true)) {
            return mod.common.throw(error.InvalidArgument, "file path is not allowed: {s}", .{file_path});
        }
        const full = try std.mem.concat(a, u8, &.{ media_path, file_path });
        defer a.free(full);
        const fz = mod.common.dupZ(full);
        defer mod.common.freeZ(fz);
        mod.common.srv(.inf, "handle_media", "loading image from local file '%s'\n", .{fz});
        // `std::ifstream(full, std::ios::binary)`, read whole
        const data = try stdlib.file.readStream(stdlib.file.defaultIo(), a, full) orelse
            return mod.common.throw(error.InvalidArgument, "file does not exist or cannot be opened: {s}", .{file_path});
        out_files.append(a, data) catch |e| {
            a.free(data);
            return e;
        };
    } else if (std.mem.startsWith(u8, url, "data:")) {
        // try to decode base64 image, video, or audio
        const n_parts = std.mem.count(u8, url, ",") + 1;
        if (n_parts != 2) {
            return mod.common.throw(error.InvalidArgument, "Invalid uri-encoded base64 value", .{});
        }
        const comma = std.mem.indexOfScalar(u8, url, ',').?;
        const part0 = url[0..comma];
        const part1 = url[comma + 1 ..];
        if (!std.mem.startsWith(u8, part0, "data:image/") and
            !std.mem.startsWith(u8, part0, "data:video/") and
            !std.mem.startsWith(u8, part0, "data:audio/"))
        {
            return mod.common.throw(error.InvalidArgument, "Invalid uri format: {s}", .{part0});
        } else if (!std.mem.endsWith(u8, part0, "base64")) {
            return mod.common.throw(error.InvalidArgument, "uri must be base64 encoded", .{});
        } else {
            const decoded_data = try mod.tokens.base64Decode(a, part1);
            out_files.append(a, decoded_data) catch |e| {
                a.free(decoded_data);
                return e;
            };
        }
    } else {
        // try as raw base64 string
        const decoded_data = try mod.tokens.base64Decode(a, url);
        if (decoded_data.len == 0) {
            a.free(decoded_data);
            return mod.common.throw(error.RuntimeError, "Invalid base64 value", .{});
        }
        out_files.append(a, decoded_data) catch |e| {
            a.free(decoded_data);
            return e;
        };
    }
}

// -----------------------------------------------------------------------------
// chat templates

/// Ports `common_chat_templates` (chat.cpp:314 @b1b770d90), through
/// `cli/common_chat/templates.zig`: the model's templates (or the
/// override) parsed by the Jinja engine, which both arms render from.
pub const Templates = chat.templates.Templates;

/// `std::map<std::string, std::string>`: sorted by key, strings owned by
/// the map's allocator.
pub const KwMap = struct {
    pub const Entry = struct { key: []const u8, value: []const u8 };
    entries: std.ArrayList(Entry) = .empty,

    /// `map[key] = value`.
    pub fn put(self: *KwMap, a: Allocator, key: []const u8, value: []const u8) mod.common.Error!void {
        const v = try a.dupe(u8, value);
        errdefer a.free(v);
        var i: usize = 0;
        while (i < self.entries.items.len) : (i += 1) {
            const e = &self.entries.items[i];
            switch (std.mem.order(u8, e.key, key)) {
                .lt => {},
                .eq => {
                    a.free(e.value);
                    e.value = v;
                    return;
                },
                .gt => break,
            }
        }
        const k = try a.dupe(u8, key);
        errdefer a.free(k);
        try self.entries.insert(a, i, .{ .key = k, .value = v });
    }

    /// `map.find(key)`.
    pub fn get(self: *const KwMap, key: []const u8) ?[]const u8 {
        for (self.entries.items) |e| if (std.mem.eql(u8, e.key, key)) return e.value;
        return null;
    }

    /// `map.erase(key)`.
    pub fn erase(self: *KwMap, a: Allocator, key: []const u8) void {
        for (self.entries.items, 0..) |e, i| if (std.mem.eql(u8, e.key, key)) {
            a.free(e.key);
            a.free(e.value);
            _ = self.entries.orderedRemove(i);
            return;
        };
    }

    /// A copy owned by `a`.
    pub fn clone(self: *const KwMap, a: Allocator) mod.common.Error!KwMap {
        var out: KwMap = .{};
        errdefer out.deinit(a);
        for (self.entries.items) |e| try out.put(a, e.key, e.value);
        return out;
    }

    /// The entries as `templates.zig` takes them, borrowed from the map.
    ///
    /// Return: owned by `a`.
    pub fn kwArgs(self: *const KwMap, a: Allocator) mod.common.Error![]const chat.templates.KwArg {
        const out = try a.alloc(chat.templates.KwArg, self.entries.items.len);
        for (self.entries.items, out) |e, *o| o.* = .{ .key = e.key, .value = e.value };
        return out;
    }

    pub fn deinit(self: *KwMap, a: Allocator) void {
        for (self.entries.items) |e| {
            a.free(e.key);
            a.free(e.value);
        }
        self.entries.deinit(a);
    }
};

/// Ports `common_chat_templates_inputs` (chat.h:262 @b1b770d90), as the
/// conversion builds it: lists it grows and the kwargs map it merges into.
/// The messages and tools are owned by the allocator the conversion used.
pub const ChatTemplatesInputs = struct {
    messages: std.ArrayList(chat.Msg) = .empty,
    grammar: []const u8 = "",
    json_schema: []const u8 = "",
    add_generation_prompt: bool = true,
    continue_final_message: chat.Continuation = .none,
    use_jinja: bool = true,
    // Parameters below only supported when use_jinja is true
    tools: std.ArrayList(chat.Tool) = .empty,
    tool_choice: chat.ToolChoice = .auto,
    parallel_tool_calls: bool = false,
    reasoning_format: chat.ReasoningFormat = .none, // TODO: refactor this to "bool enable_thinking"
    enable_thinking: bool = true,
    chat_template_kwargs: KwMap = .{},
    add_bos: bool = false,
    add_eos: bool = false,
    force_pure_content: bool = false,
};

/// Ports `common_chat_params` (chat.h:304 @b1b770d90): `templates.zig`'s.
/// Every string and list is owned by the allocator `chatTemplatesApply`
/// took.
pub const ChatParams = chat.params.ChatParams;

/// `json_schema_to_grammar(schema)`, its `std::invalid_argument` text moved
/// into the data layer's `what()`.
///
/// Return: owned by `a`.
pub fn schemaToGrammar(a: Allocator, schema: *const json.Value) mod.common.Error![]u8 {
    return json.schema_to_grammar.jsonSchemaToGrammar(a, schema, false) catch |e| switch (e) {
        error.InvalidArgument, error.RuntimeError => mod.common.throw(e, "{s}", .{json.schema.what()}),
        error.JsonError => error.JsonError,
        error.OutOfMemory => error.OutOfMemory,
    };
}

/// Ports `common_chat_templates_apply` (chat.cpp:1698 @b1b770d90): the
/// conversation rendered through Jinja (`--jinja`, with the parsers and
/// grammars the template's analysis gives) or through libllama's built-in
/// template families (`--no-jinja`), both `templates.zig`'s.
///
/// Return: owned by `a`, which must be an arena.
pub fn chatTemplatesApply(a: Allocator, tmpls: *const Templates, inputs: *const ChatTemplatesInputs) mod.common.Error!ChatParams {
    const in = try chatTemplatesInputs(a, inputs);
    return chat.templates.apply(a, tmpls, &in);
}

/// `common_chat_templates_inputs` as `templates.zig` takes it.
///
/// Return: borrows `inputs`; the kwargs are `a`'s.
fn chatTemplatesInputs(a: Allocator, inputs: *const ChatTemplatesInputs) mod.common.Error!chat.templates.TemplatesInputs {
    return .{
        .messages = inputs.messages.items,
        .grammar = inputs.grammar,
        .json_schema = inputs.json_schema,
        .add_generation_prompt = inputs.add_generation_prompt,
        .continue_final_message = inputs.continue_final_message,
        .use_jinja = inputs.use_jinja,
        .tools = inputs.tools.items,
        .tool_choice = inputs.tool_choice,
        .parallel_tool_calls = inputs.parallel_tool_calls,
        .reasoning_format = inputs.reasoning_format,
        .enable_thinking = inputs.enable_thinking,
        .chat_template_kwargs = try inputs.chat_template_kwargs.kwArgs(a),
        .add_bos = inputs.add_bos,
        .add_eos = inputs.add_eos,
        .force_pure_content = inputs.force_pure_content,
    };
}

/// `caps["supports_parallel_tool_calls"]` of `common_chat_templates_get_caps`
/// (`templates.getCaps`): `caps[key]` of a missing key is false.
pub fn chatTemplatesGetCaps(tmpls: *const Templates) bool {
    return chat.templates.capsValue(chat.templates.getCaps(tmpls), "supports_parallel_tool_calls");
}

// -----------------------------------------------------------------------------
// /chat/completions

/// Ports `server_chat_params` (server-common.h:363 @b1b770d90): the global
/// parameters for chat formatting and parsing.
pub const ServerChatParams = struct {
    use_jinja: bool,
    prefill_assistant: bool,
    reasoning_format: chat.ReasoningFormat,
    chat_template_kwargs: KwMap = .{}, // mapping key --> json value
    tmpls: *const Templates,
    allow_image: bool,
    allow_audio: bool,
    allow_video: bool,
    enable_thinking: bool = true,
    reasoning_budget: c_int = -1,
    reasoning_budget_message: []const u8 = "",
    media_path: []const u8 = "",
    media_allow_private_urls: bool = false,
    force_pure_content: bool = false,
};

/// Ports `oaicompat_chat_params_parse` (server-common.cpp:1413 @b1b770d90): an
/// OpenAI `/chat/completions` body as llama.cpp's own parameters -- the
/// conversation rendered into `prompt`, every media part decoded into
/// `out_files` and replaced in `body` by the media marker -- and the chat
/// session that rendered it, which owns the grammar, triggers and parser
/// the server applies to each task (`Task.applyChatSession`).
///
/// Parameters:
/// - `body`: the request, owned by `a`; its messages' media parts are
///   rewritten in place.
/// - `out_files`: grows; each file owned by `a`.
///
/// Return: owned by `a`.
/// - `out_session`: replaced by the request's session, owned by the
///   session's allocator; the caller deinits it.
pub fn oaicompatChatParamsParse(a: Allocator, vocab: ?*const mod.common.c.llama_vocab, body: *json.Value, opt: *const ServerChatParams, out_files: *std.ArrayList([]u8), out_session: *chat.session.Session) mod.common.Error!json.Value {
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();

    var llama_params: json.Value = .null;
    errdefer llama_params.deinit(a);

    var tools_v = try mod.common.jsonValueJson(ar, body, "tools", .null);
    const has_tools = tools_v.isArray() and !tools_v.empty();
    const stream = mod.common.jsonValue(body, "stream", bool, false);
    const tool_choice = mod.common.jsonValue(body, "tool_choice", []const u8, "auto");

    if (!opt.use_jinja) {
        if (has_tools) {
            return mod.common.throw(error.RuntimeError, "tools param requires --jinja flag", .{});
        }
        if (!std.mem.eql(u8, tool_choice, "auto")) {
            return mod.common.throw(error.RuntimeError, "tool_choice param requires --jinja flag", .{});
        }
    }

    try handleStop(a, &llama_params, body);

    var json_schema = try mod.common.jsonValueJson(ar, body, "json_schema", .null);
    const grammar = mod.common.jsonValue(body, "grammar", []const u8, "");
    if (!json_schema.isNull() and grammar.len != 0) {
        return mod.common.throw(error.RuntimeError, "Cannot use both json_schema and grammar", .{});
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        const response_format = try mod.common.jsonValueJson(ar, body, "response_format", json.Value.emptyObject());
        const response_type = mod.common.jsonValue(&response_format, "type", []const u8, "");
        if (std.mem.eql(u8, response_type, "json_object")) {
            if (response_format.contains("schema") or json_schema.empty()) {
                json_schema = try mod.common.jsonValueJson(ar, &response_format, "schema", json.Value.emptyObject());
            }
        } else if (std.mem.eql(u8, response_type, "json_schema")) {
            if (response_format.contains("json_schema")) {
                const schema_wrapper = try mod.common.jsonValueJson(ar, &response_format, "json_schema", json.Value.emptyObject());
                json_schema = try mod.common.jsonValueJson(ar, &schema_wrapper, "schema", json.Value.emptyObject());
            } else {
                json_schema = try mod.common.jsonValueJson(ar, &response_format, "schema", json.Value.emptyObject());
            }
        } else if (response_type.len != 0 and !std.mem.eql(u8, response_type, "text")) {
            return mod.common.throw(error.InvalidArgument, "response_format type must be one of \"text\", \"json_object\" or \"json_schema\", but got: {s}", .{response_type});
        }
    }

    // an absent or empty schema means any object
    if (json_schema.isObject() and json_schema.empty()) {
        try json_schema.set(ar, "type", try json.Value.initString(ar, "object"));
    }

    // get input files
    if (!body.contains("messages")) {
        return mod.common.throw(error.InvalidArgument, "'messages' is required", .{});
    }
    const messages = try body.at("messages");
    if (!messages.isArray()) {
        return mod.common.throw(error.InvalidArgument, "Expected 'messages' to be an array", .{});
    }
    for (messages.array.items) |*msg| {
        const role = mod.common.jsonValue(msg, "role", []const u8, "");
        if (!std.mem.eql(u8, role, "assistant") and !msg.contains("content")) {
            return mod.common.throw(error.InvalidArgument, "All non-assistant messages must contain 'content'", .{});
        }
        if (std.mem.eql(u8, role, "assistant")) {
            if (!msg.contains("content") and !msg.contains("tool_calls")) {
                return mod.common.throw(error.InvalidArgument, "Assistant message must contain either 'content' or 'tool_calls'!", .{});
            }
            if (!msg.contains("content")) {
                continue; // avoid errors with no content
            }
        }
        const content = try msg.at("content");
        if (content.isString() or content.isNull()) {
            continue;
        }

        if (!content.isArray()) {
            return mod.common.throw(error.InvalidArgument, "Expected 'content' to be a string or an array", .{});
        }

        try oaicompatContentLoadMedia(a, ar, content, opt, out_files);
    }

    const caps_parallel = chatTemplatesGetCaps(opt.tmpls);

    var inputs: ChatTemplatesInputs = .{};
    inputs.messages = try chat.msg.msgsParseOaicompat(ar, messages);
    inputs.tools = try chat.msg.toolsParseOaicompat(ar, &tools_v);
    inputs.tool_choice = try chat.msg.toolChoiceParseOaicompat(tool_choice);
    inputs.json_schema = if (json_schema.isNull()) "" else try json.dump(ar, &json_schema, -1);
    inputs.grammar = grammar;
    inputs.use_jinja = opt.use_jinja;
    inputs.parallel_tool_calls = mod.common.jsonValue(body, "parallel_tool_calls", bool, caps_parallel);
    inputs.add_generation_prompt = mod.common.jsonValue(body, "add_generation_prompt", bool, true);
    inputs.continue_final_message = if (body.contains("continue_final_message"))
        try chat.msg.continuationParse(try body.at("continue_final_message"))
    else
        .none;
    const msgs = inputs.messages.items;
    if (inputs.continue_final_message == .none and opt.prefill_assistant and
        msgs.len != 0 and std.mem.eql(u8, msgs[msgs.len - 1].role, "assistant"))
    {
        if (msgs.len >= 2 and std.mem.eql(u8, msgs[msgs.len - 2].role, "assistant")) {
            return mod.common.throw(error.InvalidArgument, "Cannot have 2 or more assistant messages at the end of the list.", .{});
        }
        inputs.continue_final_message = .auto;
        inputs.add_generation_prompt = false;
    }
    if (inputs.continue_final_message != .none and inputs.add_generation_prompt) {
        return mod.common.throw(error.InvalidArgument, "Cannot set both add_generation_prompt and continue_final_message to true.", .{});
    }
    if (inputs.continue_final_message != .none and
        msgs.len != 0 and
        std.mem.eql(u8, msgs[msgs.len - 1].role, "assistant") and
        msgs[msgs.len - 1].tool_calls.items.len != 0)
    {
        return mod.common.throw(error.InvalidArgument, "Cannot continue an assistant message that contains tool calls.", .{});
    }
    inputs.reasoning_format = opt.reasoning_format;
    if (body.contains("reasoning_format")) {
        inputs.reasoning_format = try chat.ReasoningFormat.fromName(try (try body.at("reasoning_format")).get([]const u8));
    }
    inputs.enable_thinking = opt.enable_thinking;
    if (inputs.tools.items.len != 0 and inputs.tool_choice != .none) {
        if (body.contains("grammar")) {
            return mod.common.throw(error.InvalidArgument, "Cannot use custom grammar constraints with tools.", .{});
        }
    }

    // merge the template args provided from command line with the args provided in the user request
    var chat_template_kwargs_object = try mod.common.jsonValueJson(ar, body, "chat_template_kwargs", json.Value.emptyObject());
    inputs.chat_template_kwargs = try opt.chat_template_kwargs.clone(ar);
    var kit = chat_template_kwargs_object.items();
    while (kit.next()) |item| {
        try inputs.chat_template_kwargs.put(ar, item.key, try json.dump(ar, item.value, -1));
    }

    // parse the "enable_thinking" kwarg to override the default value
    const enable_thinking_kwarg = inputs.chat_template_kwargs.get("enable_thinking") orelse "";
    if (std.mem.eql(u8, enable_thinking_kwarg, "true")) {
        inputs.enable_thinking = true;
    } else if (std.mem.eql(u8, enable_thinking_kwarg, "false")) {
        inputs.enable_thinking = false;
    } else if (enable_thinking_kwarg.len != 0 and enable_thinking_kwarg[0] == '"') {
        return mod.common.throw(error.InvalidArgument, "invalid type for \"enable_thinking\" (expected boolean, got string)", .{});
    }

    // Parse the OAI "reasoning_effort" field; "none" disables reasoning.
    if (body.contains("reasoning_effort")) {
        const reasoning_effort = mod.common.jsonValue(body, "reasoning_effort", []const u8, "");
        if (std.mem.eql(u8, reasoning_effort, "none")) {
            inputs.enable_thinking = false;
            inputs.chat_template_kwargs.erase(ar, "reasoning_effort");
        } else if (reasoning_effort.len != 0) {
            var s: json.Value = .{ .string = @constCast(reasoning_effort) };
            try inputs.chat_template_kwargs.put(ar, "reasoning_effort", try json.dump(ar, &s, -1));
        }
    }

    inputs.force_pure_content = opt.force_pure_content;

    // Apply chat template to the list of messages
    {
        const in = try chatTemplatesInputs(ar, &inputs);
        const session = try chat.session.Session.init(out_session.gpa, opt.tmpls, vocab, &in, .{ .echo = mod.common.jsonValue(body, "echo", bool, false) });
        out_session.deinit();
        out_session.* = session;
    }

    try llama_params.set(a, "prompt", try json.Value.initString(a, try out_session.prompt().str(ar)));

    // Reasoning budget: pass parameters through to sampling layer
    {
        const thinking = mod.common.jsonValue(body, "thinking_budget_tokens", c_int, -1);
        var reasoning_budget = mod.common.jsonValue(body, "reasoning_budget_tokens", c_int, thinking);
        if (reasoning_budget == -1) {
            reasoning_budget = opt.reasoning_budget;
        }

        if (out_session.thinkingEndTags().len != 0) {
            try llama_params.set(a, "reasoning_budget_tokens", try json.Value.from(a, reasoning_budget));
            try llama_params.set(a, "reasoning_budget_start_tag", try json.Value.initString(a, out_session.thinkingStartTag()));
            try llama_params.set(a, "reasoning_budget_end_tags", try mod.common.from(a, out_session.thinkingEndTags()));
            try llama_params.set(a, "reasoning_budget_message", try json.Value.initString(a, mod.common.jsonValue(body, "reasoning_budget_message", []const u8, opt.reasoning_budget_message)));
            try llama_params.set(a, "reasoning_control", .{ .boolean = mod.common.jsonValue(body, "reasoning_control", bool, false) });
        }
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (mod.common.jsonValue(body, "logprobs", bool, false)) {
        if (has_tools and stream) {
            return mod.common.throw(error.InvalidArgument, "logprobs is not supported with tools + stream", .{});
        }
        try llama_params.set(a, "n_probs", try json.Value.from(a, mod.common.jsonValue(body, "top_logprobs", c_int, 20)));
    } else if (body.contains("top_logprobs") and !(try body.at("top_logprobs")).isNull()) {
        return mod.common.throw(error.InvalidArgument, "top_logprobs requires logprobs to be set to true", .{});
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    try copyRemaining(a, &llama_params, body);

    // the session owns these, the server applies them with server_task::apply_chat_session()
    for ([_][]const u8{ "grammar_lazy", "grammar_triggers", "preserved_tokens" }) |key| {
        try llama_params.erase(a, key);
    }
    if (out_session.grammar().len != 0) {
        try llama_params.erase(a, "grammar");
        try llama_params.erase(a, "json_schema");
    }

    return llama_params;
}

/// Ports `oaicompat_content_load_media` (server-common.cpp:1339 @b1b770d90):
/// load the media files of an OAI content array, then replace each media
/// part with a media marker text part.
///
/// Parameters:
/// - `content`: an array, owned by `a`; rewritten in place.
/// - `ar`: scratch for the parts' sub-objects.
/// - `out_files`: grows; each file owned by `a`.
fn oaicompatContentLoadMedia(a: Allocator, ar: Allocator, content: *json.Value, opt: *const ServerChatParams, out_files: *std.ArrayList([]u8)) mod.common.Error!void {
    for (content.array.items) |*p| {
        const ty = mod.common.jsonValue(p, "type", []const u8, "");
        if (std.mem.eql(u8, ty, "image_url")) {
            if (!opt.allow_image) {
                return mod.common.throw(error.RuntimeError, "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj", .{});
            }

            const image_url = try mod.common.jsonValueJson(ar, p, "image_url", json.Value.emptyObject());
            const url = mod.common.jsonValue(&image_url, "url", []const u8, "");
            try handleMedia(a, out_files, url, opt.media_path, opt.media_allow_private_urls);

            try setMediaMarker(a, p, "image_url");
        } else if (std.mem.eql(u8, ty, "input_audio")) {
            if (!opt.allow_audio) {
                return mod.common.throw(error.RuntimeError, "audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj", .{});
            }

            // note: don't need to validate "format", it's redundant
            const input_audio = try mod.common.jsonValueJson(ar, p, "input_audio", json.Value.emptyObject());
            const url = mod.common.jsonValue(&input_audio, "data", []const u8, mod.common.jsonValue(&input_audio, "url", []const u8, ""));
            try handleMedia(a, out_files, url, opt.media_path, opt.media_allow_private_urls);

            try setMediaMarker(a, p, "input_audio");
        } else if (std.mem.eql(u8, ty, "input_video") or std.mem.eql(u8, ty, "video_url")) {
            if (!opt.allow_video) {
                return mod.common.throw(error.RuntimeError, "video input is not supported - hint: if this is unexpected, you may need to provide the mmproj", .{});
            }

            // accept the OpenAI-style "video_url" key as an alias of "input_video"
            const input_video = try mod.common.jsonValueJson(ar, p, ty, json.Value.emptyObject());
            const url = mod.common.jsonValue(&input_video, "data", []const u8, mod.common.jsonValue(&input_video, "url", []const u8, ""));
            try handleMedia(a, out_files, url, opt.media_path, opt.media_allow_private_urls);

            try setMediaMarker(a, p, "input_video");
            try p.erase(a, "video_url");
        } else if (!std.mem.eql(u8, ty, "text") and !std.mem.eql(u8, ty, "tool_reference")) {
            return mod.common.throw(error.InvalidArgument, "unsupported content[].type", .{});
        }
    }
}

/// Ports `tokenize_oai_content_array` (server-common.cpp:1392 @b1b770d90):
/// an `/embeddings` input in the shape of a chat message's content array --
/// the text parts concatenated, the media parts through
/// `process_mtmd_prompt`.
///
/// Return: owned by `gpa`.
pub fn tokenizeOaiContentArray(
    gpa: Allocator,
    vocab: ?*const mod.common.c.llama_vocab,
    mctx: ?*mod.common.c.mtmd_context,
    opt: *const ServerChatParams,
    content_in: *const json.Value,
    add_special: bool,
    parse_special: bool,
    init_opt: mod.common.c.mtmd_helper_init_opt,
) mod.common.Error!mod.tokens.ServerTokens {
    if (!content_in.isArray()) {
        return mod.common.throw(error.InvalidArgument, "\"content\" must be an array", .{});
    }

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const ar = arena_state.allocator();

    var content = try content_in.clone(ar);
    var files: std.ArrayList([]u8) = .empty;
    try oaicompatContentLoadMedia(ar, ar, &content, opt, &files);

    var prompt: std.ArrayList(u8) = .empty;
    for (content.array.items) |*p| {
        try prompt.appendSlice(ar, mod.common.jsonValue(p, "text", []const u8, ""));
    }

    if (files.items.len == 0) {
        const toks = try mod.common.tokenize(ar, vocab, prompt.items, add_special, parse_special);
        return mod.tokens.ServerTokens.fromTokens(gpa, toks, false);
    }
    const fs: []const []const u8 = @ptrCast(files.items);
    return mod.tokens.processMtmdPrompt(gpa, mctx, prompt.items, fs, init_opt, false);
}

/// `p["type"] = "media_marker"; p["text"] = get_media_marker(); p.erase(key);`
fn setMediaMarker(a: Allocator, p: *json.Value, key: []const u8) mod.common.Error!void {
    try p.set(a, "type", try json.Value.initString(a, "media_marker"));
    try p.set(a, "text", try json.Value.initString(a, mod.common.getMediaMarker()));
    try p.erase(a, key);
}

// -----------------------------------------------------------------------------
// responses

/// Ports the `encode` overload over a buffer (base64.hpp:171 @b1b770d90): the
/// standard alphabet, padded.
///
/// Return: owned by `a`.
pub fn base64Encode(a: Allocator, data: []const u8) mod.common.Error![]u8 {
    const enc = std.base64.standard.Encoder;
    const out = try a.alloc(u8, enc.calcSize(data.len));
    _ = enc.encode(out, data);
    return out;
}

/// Ports `format_embeddings_response_oaicompat` (server-common.cpp:1636 @b1b770d90).
///
/// Return: owned by `a`.
pub fn formatEmbeddingsResponseOaicompat(a: Allocator, request: *const json.Value, model_name: []const u8, embeddings: *json.Value, use_base64: bool) mod.common.Error!json.Value {
    var data = json.Value.emptyArray();
    errdefer data.deinit(a);
    var n_tokens: i32 = 0;
    var i: c_int = 0;
    var it = embeddings.items();
    while (it.next()) |entry| {
        const elem = entry.value;
        var embedding_obj = mod.common.Obj.init(a);
        errdefer embedding_obj.deinit();

        if (use_base64) {
            var emb = try mod.common.jsonValueJson(a, elem, "embedding", json.Value.emptyArray());
            defer emb.deinit(a);
            const vec = try emb.getSlice(a, f32);
            defer a.free(vec);
            const b64 = try base64Encode(a, std.mem.sliceAsBytes(vec));
            defer a.free(b64);
            try embedding_obj.put("embedding", b64);
            try embedding_obj.put("index", i);
            i += 1;
            try embedding_obj.put("object", "embedding");
            try embedding_obj.put("encoding_format", "base64");
        } else {
            try embedding_obj.put("embedding", try mod.common.jsonValueJson(a, elem, "embedding", json.Value.emptyArray()));
            try embedding_obj.put("index", i);
            i += 1;
            try embedding_obj.put("object", "embedding");
        }
        try data.pushBack(a, embedding_obj.done());

        n_tokens +%= mod.common.jsonValue(elem, "tokens_evaluated", c_int, 0);
    }

    var res = mod.common.Obj.init(a);
    errdefer res.deinit();
    try res.put("model", mod.common.jsonValue(request, "model", []const u8, model_name));
    try res.put("object", "list");
    var usage = mod.common.Obj.init(a);
    errdefer usage.deinit();
    try usage.put("prompt_tokens", n_tokens);
    try usage.put("total_tokens", n_tokens);
    try res.put("usage", usage.done());
    try res.put("data", data);
    data = .null;

    return res.done();
}

const ScoreCtx = struct { label: []const u8 };

fn scoreGreater(ctx: ScoreCtx, x: json.Value, y: json.Value) bool {
    return mod.common.jsonValue(&x, ctx.label, f64, 0.0) > mod.common.jsonValue(&y, ctx.label, f64, 0.0);
}

/// Ports `format_response_rerank` (server-common.cpp:1682 @b1b770d90): the
/// ranks sorted by score (libc++'s `std::sort`, ties in its order), cut to
/// `top_n`; TEI's bare array or the Jina-style object.
///
/// Parameters:
/// - `texts`: the documents; a returned one is moved out (left empty), as
///   the C++ moves it.
///
/// Return: owned by `a`.
pub fn formatResponseRerank(
    a: Allocator,
    request: *const json.Value,
    model_name: []const u8,
    ranks: *json.Value,
    is_tei_format: bool,
    texts: [][]const u8,
    top_n: c_int,
) mod.common.Error!json.Value {
    var n_tokens: i32 = 0;
    const return_text = is_tei_format and mod.common.jsonValue(request, "return_text", bool, false);
    var elements: std.ArrayList(json.Value) = .empty; // Temporary vector to hold unsorted elements
    defer {
        for (elements.items) |*e| e.deinit(a);
        elements.deinit(a);
    }
    const score_label: []const u8 = if (is_tei_format) "score" else "relevance_score";
    var it = ranks.items();
    while (it.next()) |entry| {
        const rank = entry.value;
        const index = mod.common.jsonValue(rank, "index", c_int, 0);
        var elem = mod.common.Obj.init(a);
        errdefer elem.deinit();
        try elem.put("index", index);
        try elem.put(score_label, mod.common.jsonValue(rank, "score", f64, 0.0));
        n_tokens +%= mod.common.jsonValue(rank, "tokens_evaluated", c_int, 0);
        if (return_text) {
            const ui: usize = @intCast(index);
            try elem.put("text", texts[ui]);
            texts[ui] = "";
        }
        try elements.append(a, elem.done());
    }

    @import("stdlib").libcxx.sort(json.Value, elements.items, ScoreCtx{ .label = score_label }, scoreGreater);

    const keep = @min(@as(usize, @intCast(top_n)), elements.items.len);
    for (elements.items[keep..]) |*e| e.deinit(a);
    elements.shrinkRetainingCapacity(keep);
    var results = json.Value.emptyArray();
    errdefer results.deinit(a);
    for (elements.items) |e| try results.pushBack(a, e);
    elements.clearRetainingCapacity();

    if (is_tei_format) return results;

    var res = mod.common.Obj.init(a);
    errdefer res.deinit();
    try res.put("model", mod.common.jsonValue(request, "model", []const u8, model_name));
    try res.put("object", "list");
    var usage = mod.common.Obj.init(a);
    errdefer usage.deinit();
    try usage.put("prompt_tokens", n_tokens);
    try usage.put("total_tokens", n_tokens);
    try res.put("usage", usage.done());
    try res.put("results", results);
    results = .null;

    return res.done();
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "fs_validate_filename" {
    try std.testing.expect(fsValidateFilename("a/b.png", true));
    try std.testing.expect(!fsValidateFilename("a/b.png", false));
    try std.testing.expect(!fsValidateFilename("../x", true));
    try std.testing.expect(!fsValidateFilename("x.", true));
    try std.testing.expect(!fsValidateFilename("", true));
}

test "a /completions body keeps n_predict over max_tokens" {
    const a = std.testing.allocator;
    var body = try json.parse(a, "{\"prompt\":\"x\",\"stop\":\"s\",\"max_tokens\":3}");
    defer body.deinit(a);
    var out = try oaicompatCompletionParamsParse(a, &body);
    defer out.deinit(a);
    const s = try json.dump(a, &out, -1);
    defer a.free(s);
    try std.testing.expectEqualStrings("{\"stop\":[\"s\"],\"prompt\":\"x\",\"max_tokens\":3}", s);
}
