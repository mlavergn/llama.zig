//! The server's shared helpers: `json_value`, error responses, random ids,
//! the media marker, LoRA bookkeeping, UTF-8 checks, slot statistics and
//! server metrics, token probabilities, the SSE formatters, and the pipe
//! between threads.
//!
//! # Provenance
//!
//! Ports parts of `llama.cpp/tools/server/server-common.cpp` and
//! `llama.cpp/tools/server/server-common.h` (llama.cpp v1.0.0,
//! `b1b770d90`). The rest of that pair is `tokens.zig` (`server_tokens` and
//! prompt tokenization), `oaicompat.zig` (the OpenAI request conversions and
//! response formatters) and `subproc.zig` (`server_subproc`). Each
//! declaration names what it ports and the line it began at.
//!
//! # Exceptions
//!
//! A C++ throw is an `Error` with its `what()` in
//! `cli/common_chat/exception.zig`'s per-thread buffer (`what(err)` here);
//! `error.JsonError` keeps nlohmann's text in `cli/json`'s. The server's
//! `ex_wrapper` answers 400 for `InvalidArgument` and 500 for the rest.
//!
//! # Randomness
//!
//! `random_string` draws from `std::random_device` into a fresh
//! `std::mt19937`, as upstream does: ids are unpredictable by design, and a
//! gate masks them (`server-task-diff` and `server-diff` both do).

const std = @import("std");
const llama = @import("llama");
const stdlib = @import("stdlib");
const json = @import("json");
const common_chat = @import("common_chat");
const tools_common = @import("tools_common");

/// `LLAMA_TOKEN_NULL`.
pub const token_null: llama.impl.llama_c.llama_token = llama.impl.llama_c.LLAMA_TOKEN_NULL;

// -----------------------------------------------------------------------------
// logging and assertions

/// `GGML_ASSERT(cond)` as a header or source file of the server states it.
/// `std::filesystem::path(p).filename()` on POSIX: what follows the last
/// `/`, all of `p` when there is none.
///
/// Return: a slice of `p`.
pub fn pathFilename(p: []const u8) []const u8 {
    const i = std.mem.findScalarLast(u8, p, '/') orelse return p;
    return p[i + 1 ..];
}

/// A NUL-terminated copy for a `%s` argument.
///
/// Parameters:
/// - `a`: owns the copy, usually a request's arena.
///
/// Return: the copy, freed with `a`.
pub fn zs(a: std.mem.Allocator, s: []const u8) std.mem.Allocator.Error![*:0]const u8 {
    return (try a.dupeSentinel(u8, s, 0)).ptr;
}

/// The levels of the `SRV_*` macros.
pub const SrvLevel = enum { dbg, trc, inf, wrn, err };

/// Ports `SRV_DBG`, `SRV_TRC`, `SRV_INF`, `SRV_WRN` and `SRV_ERR`
/// (server-common.h:35, 36, 37, 38, 39 @b1b770d90): the message
/// prefixed `srv  %12.*s: ` with the calling function's name.
pub fn srv(comptime level: SrvLevel, comptime func: []const u8, comptime fmt: []const u8, args: anytype) void {
    const f = "srv  %12.*s: " ++ fmt;
    const pre = .{ @as(c_int, 12), @as([*:0]const u8, func ++ "") };
    switch (level) {
        .dbg => tools_common.log.dbg(f, pre ++ args),
        .trc => tools_common.log.trc(f, pre ++ args),
        .inf => tools_common.log.inf(f, pre ++ args),
        .wrn => tools_common.log.wrn(f, pre ++ args),
        .err => tools_common.log.err(f, pre ++ args),
    }
}

/// A NUL-terminated copy for a `%s` argument; the C stops at the first NUL
/// as `c_str()` does. Freed with `freeZ`.
pub fn dupZ(s: []const u8) [*:0]const u8 {
    const z = std.heap.c_allocator.dupeSentinel(u8, s, 0) catch return "";
    return z.ptr;
}

/// Frees what `dupZ` returned.
pub fn freeZ(z: [*:0]const u8) void {
    const s = std.mem.span(z);
    if (s.len == 0 and z == @as([*:0]const u8, "")) return;
    std.heap.c_allocator.free(s);
}

// -----------------------------------------------------------------------------
// json_value

/// Ports `json_value` (server-common.h:45 @b1b770d90) for a scalar `T`:
/// the member `key` as `T`, or `default_value` when it is absent or null,
/// or -- with a warning -- when it does not convert.
///
/// Parameters:
/// - `body`: any value; only an object has members.
/// - `T`: `bool`, an integer or float type, or `[]const u8` (borrowed from
///   `body`).
///
/// Return: the value or the default.
pub fn jsonValue(body: *const json.Value, key: []const u8, comptime T: type, default_value: T) T {
    // Fallback null to default value
    if (body.contains(key) and !(body.at(key) catch |err| switch (err) {
        error.OutOfMemory, error.JsonError => unreachable,
    }).isNull()) {
        const v = body.at(key) catch |err| switch (err) {
            error.OutOfMemory, error.JsonError => unreachable,
        };
        return v.get(T) catch {
            wrongType(key);
            return default_value;
        };
    }
    return default_value;
}

/// `json_value(body, key, json(...))`: a copy of the member, or of
/// `default_value` (which is consumed). `get<common_json>()` cannot fail.
///
/// Return: owned by the caller.
pub fn jsonValueJson(gpa: std.mem.Allocator, body: *const json.Value, key: []const u8, default_value: json.Value) common_chat.exception.Error!json.Value {
    var def = default_value;
    if (body.contains(key) and !(body.at(key) catch |err| switch (err) {
        error.OutOfMemory, error.JsonError => unreachable,
    }).isNull()) {
        def.deinit(gpa);
        return (body.at(key) catch |err| switch (err) {
            error.OutOfMemory, error.JsonError => unreachable,
        }).clone(gpa);
    }
    return def;
}

/// `json_value(body, key, std::vector<std::string>())`.
///
/// Return: owned by the caller (the list; the strings are borrowed from
/// `body`).
pub fn jsonValueStrings(gpa: std.mem.Allocator, body: *const json.Value, key: []const u8) common_chat.exception.Error![]const []const u8 {
    if (body.contains(key) and !(body.at(key) catch |err| switch (err) {
        error.OutOfMemory, error.JsonError => unreachable,
    }).isNull()) {
        const v = body.at(key) catch |err| switch (err) {
            error.OutOfMemory, error.JsonError => unreachable,
        };
        return v.getSlice(gpa, []const u8) catch |e| switch (e) {
            error.JsonError => {
                wrongType(key);
                return &.{};
            },
            else => |x| x,
        };
    }
    return &.{};
}

fn wrongType(key: []const u8) void {
    const k = dupZ(key);
    defer freeZ(k);
    const w = dupZ(json.what());
    defer freeZ(w);
    tools_common.log.wrn("Wrong type supplied for parameter '%s', using default value: %s\n", .{ k, w });
}

// -----------------------------------------------------------------------------
// building JSON

/// A `std::vector<unsigned char>`, which `common_json` makes an array of
/// unsigned numbers.
pub const Bytes = struct { s: []const u8 };

/// `common_json(x)` for the C++ types the server builds JSON from: what
/// `Value.from` takes, plus `std::vector<T>` and `std::set<T>` of numbers
/// and strings (as slices), `Bytes`, and `Value` itself (moved in).
///
/// Return: owned by the caller.
pub fn from(a: std.mem.Allocator, x: anytype) common_chat.exception.Error!json.Value {
    const T = @TypeOf(x);
    if (T == json.Value) return x;
    if (T == Bytes) {
        var arr = json.Value.emptyArray();
        errdefer arr.deinit(a);
        for (x.s) |b| try arr.pushBack(a, .{ .number_unsigned = b });
        return arr;
    }
    switch (@typeInfo(T)) {
        .pointer => |p| {
            const Elem = switch (p.size) {
                .slice => p.child,
                .one => switch (@typeInfo(p.child)) {
                    .array => |arr| arr.child,
                    else => @compileError("not a JSON value"),
                },
                else => @compileError("not a JSON value"),
            };
            if (Elem == u8) return json.Value.from(a, x);
            var arr = json.Value.emptyArray();
            errdefer arr.deinit(a);
            for (x) |e| try arr.pushBack(a, try from(a, e));
            return arr;
        },
        else => return json.Value.from(a, x),
    }
}

/// A JSON object under construction, members in insertion order.
pub const Obj = struct {
    a: std.mem.Allocator,
    v: json.Value = json.Value.emptyObject(),

    pub fn init(a: std.mem.Allocator) Obj {
        return .{ .a = a };
    }

    /// `obj[key] = x`.
    pub fn put(self: *Obj, key: []const u8, x: anytype) common_chat.exception.Error!void {
        try self.v.set(self.a, key, try from(self.a, x));
    }

    /// Return: the object, owned by the caller.
    pub fn done(self: *Obj) json.Value {
        const v = self.v;
        self.v = .null;
        return v;
    }

    pub fn deinit(self: *Obj) void {
        self.v.deinit(self.a);
    }
};

// -----------------------------------------------------------------------------
// errors

/// Ports `error_type` (server-common.h:60 @b1b770d90).
pub const ErrorType = enum(c_int) {
    invalid_request,
    authentication,
    server,
    not_found,
    permission,
    unavailable, // custom error
    not_supported, // custom error
    exceed_context_size, // custom error
};

/// Ports `format_error_response` (server-common.cpp:97 @b1b770d90).
///
/// Return: `{"code", "message", "type"}`, owned by the caller.
pub fn formatErrorResponse(a: std.mem.Allocator, message: []const u8, ty: ErrorType) common_chat.exception.Error!json.Value {
    var type_str: []const u8 = "";
    var code: c_int = 500;
    switch (ty) {
        .invalid_request => {
            type_str = "invalid_request_error";
            code = 400;
        },
        .authentication => {
            type_str = "authentication_error";
            code = 401;
        },
        .not_found => {
            type_str = "not_found_error";
            code = 404;
        },
        .server => {
            type_str = "server_error";
            code = 500;
        },
        .permission => {
            type_str = "permission_error";
            code = 403;
        },
        .not_supported => {
            type_str = "not_supported_error";
            code = 501;
        },
        .unavailable => {
            type_str = "unavailable_error";
            code = 503;
        },
        .exceed_context_size => {
            type_str = "exceed_context_size_error";
            code = 400;
        },
    }
    var o = Obj.init(a);
    errdefer o.deinit();
    try o.put("code", code);
    try o.put("message", message);
    try o.put("type", type_str);
    return o.done();
}

// -----------------------------------------------------------------------------
// grammar triggers

/// Ports `common_grammar_trigger_type` (common/common.h:148 @b1b770d90). The
/// server casts any integer to it, so it is kept as one.
pub const trigger_type_token: c_int = 0;
pub const trigger_type_word: c_int = 1;
pub const trigger_type_pattern: c_int = 2;
pub const trigger_type_pattern_full: c_int = 3;

/// Ports `common_grammar_trigger` (common/common.h:155 @b1b770d90), as
/// `server_grammar_trigger` (server-common.h:72 @b1b770d90) wraps it.
pub const GrammarTrigger = struct {
    type: c_int = 0,
    value: []const u8 = "",
    token: llama.impl.llama_c.llama_token = token_null,

    /// Ports `server_grammar_trigger`'s constructor from JSON
    /// (server-common.h:72 @b1b770d90).
    ///
    /// Return: `value` is `a`'s.
    pub fn fromJson(a: std.mem.Allocator, in: *const json.Value) common_chat.exception.Error!GrammarTrigger {
        var t: GrammarTrigger = .{};
        t.type = try (try in.at("type")).get(c_int);
        t.value = try a.dupe(u8, try (try in.at("value")).get([]const u8));
        if (t.type == trigger_type_token) {
            t.token = try (try in.at("token")).get(c_int);
        }
        return t;
    }

    /// Ports `server_grammar_trigger`'s `to_json` (server-common.h:85 @b1b770d90).
    ///
    /// Return: owned by the caller.
    pub fn toJson(self: GrammarTrigger, a: std.mem.Allocator) common_chat.exception.Error!json.Value {
        var o = Obj.init(a);
        errdefer o.deinit();
        try o.put("type", self.type);
        try o.put("value", self.value);
        if (self.type == trigger_type_token) {
            try o.put("token", self.token);
        }
        return o.done();
    }
};

// -----------------------------------------------------------------------------
// random string / id

/// Ports `random_string` (server-common.cpp:172 @b1b770d90): 32 characters of
/// `[0-9A-Za-z]` from an `mt19937` seeded by the random device.
///
/// Return: owned by the caller.
pub fn randomString(a: std.mem.Allocator) common_chat.exception.Error![]u8 {
    const str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    var generator = stdlib.libcxx.Mt19937.init(stdlib.libcxx.randomDevice());
    const result = try a.alloc(u8, 32);
    for (result) |*ch| ch.* = str[generator.next() % str.len];
    return result;
}

/// Ports `gen_chatcmplid` (server-common.cpp:187 @b1b770d90).
pub fn genChatcmplid(a: std.mem.Allocator) common_chat.exception.Error![]u8 {
    const r = try randomString(a);
    defer a.free(r);
    return std.mem.concat(a, u8, &.{ "chatcmpl-", r });
}

/// Ports `gen_tool_call_id` (server-common.cpp:191 @b1b770d90).
pub fn genToolCallId(a: std.mem.Allocator) common_chat.exception.Error![]u8 {
    return randomString(a);
}

var media_marker_mutex: stdlib.sync.Mutex = .{};
var media_marker_done = false;
var media_marker_buf: [64]u8 = undefined;
var media_marker: [:0]const u8 = "";

fn initMediaMarker() void {
    // allow user to pin a reproducible marker via env var
    if (stdlib.env.get("LLAMA_MEDIA_MARKER")) |env| {
        if (env.len != 0) {
            media_marker = std.heap.c_allocator.dupeSentinel(u8, env, 0) catch "";
            return;
        }
    }
    var r: [32]u8 = undefined;
    const s = randomString(std.heap.c_allocator) catch return;
    @memcpy(&r, s);
    std.heap.c_allocator.free(s);
    media_marker = std.fmt.bufPrintSentinel(&media_marker_buf, "<__media_{s}__>", .{r}, 0) catch "";
}

/// Ports `get_media_marker` (server-common.cpp:195 @b1b770d90): fixed for the
/// process's life, from `LLAMA_MEDIA_MARKER` when set.
pub fn getMediaMarker() [:0]const u8 {
    // a function-local static: initialized once, thread-safe
    media_marker_mutex.lock();
    defer media_marker_mutex.unlock();
    if (!media_marker_done) {
        initMediaMarker();
        media_marker_done = true;
    }
    return media_marker;
}

// -----------------------------------------------------------------------------
// audio utils

/// Ports `pcm_f32_to_s16` (server-common.cpp:211 @b1b770d90): float samples
/// to signed 16-bit PCM bytes, clamped to [-1, 1].
///
/// Return: owned by `a`.
pub fn pcmF32ToS16(a: std.mem.Allocator, samples: []const f32) std.mem.Allocator.Error![]u8 {
    const bytes = try a.alloc(u8, samples.len * @sizeOf(i16));
    for (samples, 0..) |x, i| {
        // std::max(-1.0f, std::min(1.0f, x)), NaN clamped as the C++'s comparisons do
        const lo = if (x < 1.0) x else 1.0;
        const clamped = if (-1.0 < lo) lo else -1.0;
        // std::lround, then the narrowing cast
        const v: i16 = @intFromFloat(@round(clamped * 32767.0));
        std.mem.writeInt(i16, bytes[2 * i ..][0..2], v, .little);
    }
    return bytes;
}

// -----------------------------------------------------------------------------
// model output modalities

/// Ports `server_model_output_modalities` (server-common.cpp:224
/// @b1b770d90): output modalities for `architecture.output_modalities` in
/// GET /models.
///
/// Parameters:
/// - `has_cap_chat`: false for a tts-only model.
/// - `has_cap_tts`: the model can generate audio.
///
/// Return: static strings.
pub fn serverModelOutputModalities(decision_type: tools_common.common.DecisionType, has_cap_chat: bool, has_cap_tts: bool) []const []const u8 {
    return switch (decision_type) {
        .openjev, .lev, .kev, .nimble, .laya, .clef, .pplx_decider, .vllm_decision2, .vllm_d3, .lfm2_d1, .lfm2_d1_omni, .h2o_lightning, .minicpm_ocr_jev, .lfm3_d1, .lfm3_d1_omni => if (has_cap_tts) &.{ "decisions", "audio" } else &.{"decisions"},
        .granite_switch => if (has_cap_tts) &.{ "text", "decisions", "audio" } else &.{ "text", "decisions" },
        else => blk: {
            // a tts-only model cannot generate text
            if (has_cap_chat or !has_cap_tts) {
                // fallback when there is no decision type or the metadata is bad
                break :blk if (has_cap_tts) &.{ "text", "audio" } else &.{"text"};
            }
            break :blk &.{"audio"};
        },
    };
}

/// Ports `server_model_architecture_json` (server-common.cpp:265
/// @b1b770d90): the architecture object of GET /models; shared by the
/// direct server and the router.
///
/// Return: owned by `a`.
pub fn serverModelArchitectureJson(a: std.mem.Allocator, inp_image: bool, inp_audio: bool, inp_video: bool, output_modalities: []const []const u8) common_chat.exception.Error!json.Value {
    var input_modalities: [4][]const u8 = undefined;
    var n: usize = 0;
    input_modalities[n] = "text";
    n += 1;
    if (inp_image) {
        input_modalities[n] = "image";
        n += 1;
    }
    if (inp_audio) {
        input_modalities[n] = "audio";
        n += 1;
    }
    if (inp_video) {
        input_modalities[n] = "video";
        n += 1;
    }

    var o = Obj.init(a);
    errdefer o.deinit();
    try o.put("input_modalities", input_modalities[0..n]);
    try o.put("output_modalities", output_modalities);
    return o.done();
}

// -----------------------------------------------------------------------------
// lora utils

/// Ports `lora_all_alora` (server-common.cpp:291 @b1b770d90).
pub fn loraAllAlora(loras: []const tools_common.params.LoraInfo) bool {
    var found_alora = false;
    for (loras) |lora| {
        if (lora.scale != 0) {
            if (llama.api.adapter.llama_adapter_get_alora_n_invocation_tokens(lora.ptr) == 0) {
                return false;
            }
            found_alora = true;
        }
    }
    return found_alora;
}

/// Ports `lora_should_clear_cache` (server-common.cpp:304 @b1b770d90).
pub fn loraShouldClearCache(current: []const tools_common.params.LoraInfo, next: []const tools_common.params.LoraInfo) bool {
    // This should always be called after determining that the two sets are
    // _not_ equal. This assert is therefore some slightly wasted work and
    // should be safe to remove as long as this method is called correctly.
    tools_common.ggml.ggmlAssert(!areLoraEqual(current, next), "server-common.cpp", 311, "!are_lora_equal(current, next)");

    return (!(loraEnabledEmpty(current) or loraAllAlora(current)) or !loraAllAlora(next));
}

fn loraEnabledEmpty(loras: []const tools_common.params.LoraInfo) bool {
    for (loras) |l| if (l.scale > 0) return false;
    return true;
}

/// One entry of `std::map<int, float>`, which sorts by id.
pub const LoraScale = struct { id: c_int, scale: f32 };

/// `std::map<int, float>`: an adapter's id to its scale, sorted by id.
pub const LoraMap = struct {
    entries: std.ArrayList(LoraScale) = .empty,

    /// `lora[id] = scale`.
    pub fn put(self: *LoraMap, a: std.mem.Allocator, id: c_int, scale: f32) common_chat.exception.Error!void {
        var i: usize = 0;
        while (i < self.entries.items.len) : (i += 1) {
            const e = &self.entries.items[i];
            if (e.id == id) {
                e.scale = scale;
                return;
            }
            if (e.id > id) break;
        }
        try self.entries.insert(a, i, .{ .id = id, .scale = scale });
    }

    pub fn deinit(self: *LoraMap, a: std.mem.Allocator) void {
        self.entries.deinit(a);
    }

    /// A copy owned by `a`.
    pub fn clone(self: *const LoraMap, a: std.mem.Allocator) common_chat.exception.Error!LoraMap {
        return .{ .entries = .fromOwnedSlice(try a.dupe(LoraScale, self.entries.items)) };
    }
};

/// Ports `parse_lora_request` (server-common.cpp:318 @b1b770d90).
///
/// Return: owned by `a`.
pub fn parseLoraRequest(a: std.mem.Allocator, data: *json.Value) common_chat.exception.Error!LoraMap {
    var lora: LoraMap = .{};
    errdefer lora.deinit(a);

    // set value
    var it = data.items();
    while (it.next()) |entry| {
        const id = jsonValue(entry.value, "id", c_int, -1);
        const scale = jsonValue(entry.value, "scale", f32, 0.0);
        try lora.put(a, id, scale);
    }

    return lora;
}

/// Ports `are_lora_equal` (server-common.cpp:331 @b1b770d90).
pub fn areLoraEqual(l1: []const tools_common.params.LoraInfo, l2: []const tools_common.params.LoraInfo) bool {
    if (l1.len != l2.len) {
        return false;
    }
    for (l1, l2) |x, y| {
        // we don't check lora.path to reduce the time complexity
        if (x.scale != y.scale or x.ptr != y.ptr) {
            return false;
        }
    }
    return true;
}

/// Ports `lora_get_enabled_ids` (server-common.cpp:346 @b1b770d90).
///
/// Return: owned by `a`.
pub fn loraGetEnabledIds(a: std.mem.Allocator, loras: []const tools_common.params.LoraInfo) common_chat.exception.Error![]usize {
    var enabled_ids: std.ArrayList(usize) = .empty;
    errdefer enabled_ids.deinit(a);
    for (loras, 0..) |l, i| {
        if (l.scale > 0) {
            try enabled_ids.append(a, i);
        }
    }
    return enabled_ids.toOwnedSlice(a);
}

// -----------------------------------------------------------------------------
// json helpers

/// Ports `json_is_array_of_numbers` (server-common.cpp:948 @b1b770d90).
pub fn jsonIsArrayOfNumbers(data: *const json.Value) bool {
    if (data.isArray()) {
        for (data.array.items) |*e| {
            if (!e.isNumberInteger()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

/// Ports `json_is_array_of_mixed_numbers_strings` (server-common.cpp:960 @b1b770d90).
pub fn jsonIsArrayOfMixedNumbersStrings(data: *const json.Value) bool {
    var seen_string = false;
    var seen_number = false;
    if (data.isArray()) {
        for (data.array.items) |*e| {
            seen_string = seen_string or e.isString();
            seen_number = seen_number or e.isNumberInteger();
            if (seen_number and seen_string) {
                return true;
            }
        }
    }
    return false;
}

/// Ports `json_is_array_and_contains_numbers` (server-common.cpp:975 @b1b770d90).
pub fn jsonIsArrayAndContainsNumbers(data: *const json.Value) bool {
    if (data.isArray()) {
        for (data.array.items) |*e| {
            if (e.isNumberInteger()) {
                return true;
            }
        }
        return false;
    }
    return false;
}

/// Ports `json_get_nested_values` (server-common.cpp:987 @b1b770d90): each
/// path's value, keyed by the path, when every key along it exists.
///
/// Return: owned by `a`.
pub fn jsonGetNestedValues(a: std.mem.Allocator, paths: []const []const u8, js: *const json.Value) common_chat.exception.Error!json.Value {
    var result = json.Value.emptyObject();
    errdefer result.deinit(a);

    for (paths) |path| {
        var current: *const json.Value = js;
        var it = std.mem.splitScalar(u8, path, '/');
        var valid_path = true;
        while (it.next()) |k| {
            if (valid_path and current.isObject() and current.contains(k)) {
                current = try current.at(k);
            } else {
                valid_path = false;
            }
        }
        if (valid_path) {
            try result.set(a, path, try current.clone(a));
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// utf-8

/// Ports `validate_utf8` (server-common.cpp:1044 @b1b770d90): the length of
/// the longest prefix that does not end inside a multi-byte character.
pub fn validateUtf8(text: []const u8) usize {
    const len = text.len;
    if (len == 0) return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    var i: usize = 1;
    while (i <= 4 and i <= len) : (i += 1) {
        const ch = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((ch & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2) return len - i;
        } else if ((ch & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3) return len - i;
        } else if ((ch & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4) return len - i;
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

/// Ports `is_valid_utf8` (server-common.cpp:1887 @b1b770d90): lead bytes and
/// continuation tags only; overlong forms and surrogates pass.
pub fn isValidUtf8(str: []const u8) bool {
    var i: usize = 0;
    const end = str.len;
    while (i < end) {
        const b = str[i];
        if (b <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            i += 1;
        } else if ((b & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - i < 2 or (str[i + 1] & 0xC0) != 0x80)
                return false;
            i += 2;
        } else if ((b & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - i < 3 or (str[i + 1] & 0xC0) != 0x80 or (str[i + 2] & 0xC0) != 0x80)
                return false;
            i += 3;
        } else if ((b & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - i < 4 or (str[i + 1] & 0xC0) != 0x80 or
                (str[i + 2] & 0xC0) != 0x80 or (str[i + 3] & 0xC0) != 0x80)
                return false;
            i += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// stats and metrics

/// Ports `server_slot_stats` (server-common.h:422 @b1b770d90): shared between
/// a slot and its task results.
pub const SlotStats = struct {
    n_prompt_cached: u64 = 0,
    n_prompt_processed: u64 = 0,
    n_gen: u64 = 0,

    // speculative decoding stats
    // note: the per-position breakdown lives in server_slot, it is not needed in a task result
    n_draft_tokens: u64 = 0,
    n_draft_accepted: u64 = 0,
    n_draft_verif_steps: u64 = 0,

    // these are absolute timestamps (in us)
    // note: must be signed - they are subtracted before the later ones are set
    t_start: i64 = 0,
    t_prompt_last: i64 = 0,
    t_gen_last: i64 = 0,

    /// Ports `update_prompt_start` (server-common.h:440 @b1b770d90).
    pub fn updatePromptStart(self: *SlotStats) void {
        tools_common.ggml.ggmlAssert(self.t_start == 0, "server-common.h", 441, "t_start == 0");
        self.t_start = llama.impl.llama_c.ggml_time_us();
    }

    /// Ports `set_prompt_last` (server-common.h:444 @b1b770d90).
    pub fn setPromptLast(self: *SlotStats, t_us: i64) void {
        tools_common.ggml.ggmlAssert(self.t_start > 0, "server-common.h", 445, "t_start > 0");
        self.t_prompt_last = t_us;
    }

    /// Ports `update_prompt_last` (server-common.h:448 @b1b770d90).
    pub fn updatePromptLast(self: *SlotStats) void {
        self.setPromptLast(llama.impl.llama_c.ggml_time_us());
    }

    /// Ports `update_gen_last` (server-common.h:451 @b1b770d90).
    pub fn updateGenLast(self: *SlotStats) void {
        tools_common.ggml.ggmlAssert(self.t_prompt_last > 0, "server-common.h", 452, "t_prompt_last > 0");
        self.t_gen_last = llama.impl.llama_c.ggml_time_us();
    }

    /// Ports `t_elapsed_us` (server-common.h:457 @b1b770d90).
    pub fn tElapsedUs(self: *const SlotStats) i64 {
        return llama.impl.llama_c.ggml_time_us() - self.t_start;
    }

    /// Ports `t_prompt_ms` (server-common.h:460 @b1b770d90).
    pub fn tPromptMs(self: *const SlotStats) f64 {
        if (self.t_prompt_last == 0) {
            return 0.0; // the prompt is not processed yet
        }
        return @as(f64, @floatFromInt(self.t_prompt_last - self.t_start)) / 1000.0;
    }

    /// Ports `t_gen_us` (server-common.h:466 @b1b770d90).
    pub fn tGenUs(self: *const SlotStats) i64 {
        if (self.t_gen_last == 0) {
            return 0; // the generation is not started yet
        }
        // clamp to 1 us, the first token can land in the same us as t_prompt_last
        return @max(1, self.t_gen_last - self.t_prompt_last);
    }

    /// Ports `t_gen_ms` (server-common.h:473 @b1b770d90).
    pub fn tGenMs(self: *const SlotStats) f64 {
        return @as(f64, @floatFromInt(self.tGenUs())) / 1000.0;
    }

    /// Ports `n_gen_steps` (server-common.h:479 @b1b770d90).
    pub fn nGenSteps(self: *const SlotStats) u64 {
        return if (self.n_gen > 0) self.n_gen - 1 else 0;
    }

    /// Ports `t_prompt_per_token_ms` (server-common.h:485 @b1b770d90).
    pub fn tPromptPerTokenMs(self: *const SlotStats) f64 {
        return if (self.n_prompt_processed > 0) self.tPromptMs() / @as(f64, @floatFromInt(self.n_prompt_processed)) else 0.0;
    }

    /// Ports `t_gen_per_token_ms` (server-common.h:488 @b1b770d90).
    pub fn tGenPerTokenMs(self: *const SlotStats) f64 {
        return if (self.nGenSteps() > 0) self.tGenMs() / @as(f64, @floatFromInt(self.nGenSteps())) else 0.0;
    }

    /// Ports `n_prompt_tps` (server-common.h:491 @b1b770d90).
    pub fn nPromptTps(self: *const SlotStats) f64 {
        const t_ms = self.tPromptMs();
        return if (t_ms > 0.0) 1e3 / t_ms * @as(f64, @floatFromInt(self.n_prompt_processed)) else 0.0;
    }

    /// Ports `n_gen_tps` (server-common.h:495 @b1b770d90).
    pub fn nGenTps(self: *const SlotStats) f64 {
        const t_ms = self.tGenMs();
        return if (t_ms > 0.0) 1e3 / t_ms * @as(f64, @floatFromInt(self.nGenSteps())) else 0.0;
    }

    /// Ports `is_set` (server-common.h:501 @b1b770d90).
    pub fn isSet(self: *const SlotStats) bool {
        return self.t_start > 0;
    }

    /// Ports `server_slot_stats`'s `to_json` (server-common.cpp:145 @b1b770d90).
    ///
    /// Return: owned by `a`.
    pub fn toJson(self: *const SlotStats, a: std.mem.Allocator) common_chat.exception.Error!json.Value {
        var base = Obj.init(a);
        errdefer base.deinit();
        try base.put("cache_n", self.n_prompt_cached);

        try base.put("prompt_n", self.n_prompt_processed);
        try base.put("prompt_ms", self.tPromptMs());
        try base.put("prompt_per_token_ms", self.tPromptPerTokenMs());
        try base.put("prompt_per_second", self.nPromptTps());

        try base.put("predicted_n", self.n_gen);
        try base.put("predicted_ms", self.tGenMs());
        try base.put("predicted_per_token_ms", self.tGenPerTokenMs());
        try base.put("predicted_per_second", self.nGenTps());

        if (self.n_draft_tokens > 0) {
            try base.put("draft_n", self.n_draft_tokens);
            try base.put("draft_n_accepted", self.n_draft_accepted);
        }

        return base.done();
    }
};

/// Ports `server_metrics` (server-common.h:510 @b1b770d90): server-global and
/// cumulative.
pub const Metrics = struct {
    /// Ports `server_metrics`'s `bucket` (server-common.h:513 @b1b770d90).
    pub const Bucket = struct {
        count: u64 = 0, // number of tokens
        steps: u64 = 0, // number of decode steps,
        // this excludes first generated token (logits from prompt batch)
        time: u64 = 0, // in microseconds

        /// Ports `n_per_second` (server-common.h:520 @b1b770d90): the rate uses
        /// the decode steps, so that "free" tokens do not inflate it.
        pub fn nPerSecond(self: Bucket) f64 {
            return if (self.time > 0) @as(f64, @floatFromInt(self.steps)) / @as(f64, @floatFromInt(self.time)) * 1e6 else 0.0;
        }

        /// Ports `add` (server-common.h:524 @b1b770d90).
        pub fn add(self: *Bucket, n: u64, n_steps: u64, t_us: u64) void {
            self.count +%= n;
            self.steps +%= n_steps;
            self.time +%= t_us;
        }
    };

    t_start: i64 = 0,

    // these are reset by reset_bucket(), only the rate is read from them
    prompt_bucket: Bucket = .{},
    predict_bucket: Bucket = .{},

    // metrics below are cumulative since the server started
    prompt: Bucket = .{}, // only processed tokens, cached ones are counted separately below
    predict: Bucket = .{},

    // tokens reused from the cache need no decode, so they only have a count
    n_prompt_cached: u64 = 0,

    n_tokens_max: u64 = 0,

    n_decode: u64 = 0,
    n_busy_slots: u64 = 0,

    n_draft_tokens: u64 = 0, // Total draft tokens generated
    n_draft_accepted: u64 = 0, // Draft tokens actually accepted
    n_draft_verif_steps: u64 = 0, // Total draft token verification steps by the target model
    n_accepted_per_pos: std.ArrayList(u64) = .empty, // Accepted tokens per draft position

    /// Ports `init` (server-common.h:552 @b1b770d90).
    pub fn init(self: *Metrics) void {
        self.t_start = llama.impl.llama_c.ggml_time_us();
    }

    /// Ports `reset_bucket` (server-common.h:556 @b1b770d90).
    pub fn resetBucket(self: *Metrics) void {
        self.prompt_bucket = .{};
        self.predict_bucket = .{};
    }

    /// Ports `add_prompt` (server-common.h:561 @b1b770d90).
    pub fn addPrompt(self: *Metrics, n_tokens: u64, t_us: u64) void {
        self.prompt.add(n_tokens, n_tokens, t_us);
        self.prompt_bucket.add(n_tokens, n_tokens, t_us);
    }

    /// Ports `add_prompt_cached` (server-common.h:566 @b1b770d90).
    pub fn addPromptCached(self: *Metrics, n_tokens: u64) void {
        self.n_prompt_cached +%= n_tokens;
    }

    /// A copy whose list `a` owns, as the C++ copy constructor makes one.
    pub fn clone(self: *const Metrics, a: std.mem.Allocator) common_chat.exception.Error!Metrics {
        var m = self.*;
        m.n_accepted_per_pos = .fromOwnedSlice(try a.dupe(u64, self.n_accepted_per_pos.items));
        return m;
    }

    pub fn deinit(self: *Metrics, a: std.mem.Allocator) void {
        self.n_accepted_per_pos.deinit(a);
    }
};

// -----------------------------------------------------------------------------
// other utils

fn logitGreater(_: void, x: llama.impl.llama_c.llama_token_data, y: llama.impl.llama_c.llama_token_data) bool {
    return x.logit > y.logit;
}

/// Ports `get_token_probabilities` (server-common.cpp:1733 @b1b770d90): the
/// candidates at output `idx`, the leading `n_top` sorted by logit
/// (libc++'s `partial_sort`), with their softmax.
///
/// Return: owned by `a`.
pub fn getTokenProbabilities(a: std.mem.Allocator, ctx: ?*llama.impl.llama_c.llama_context, idx: c_int, n_top_in: usize) common_chat.exception.Error![]llama.impl.llama_c.llama_token_data {
    var n_top = n_top_in;
    const logits = llama.api.context.llama_get_logits_ith(ctx, idx).?;
    const sampled_ids: ?[*]llama.impl.llama_c.llama_token = llama.api.context.llama_get_sampled_candidates_ith(ctx, idx);

    const n_logits: c_int = @bitCast(llama.api.context.llama_get_sampled_logits_count_ith(ctx, idx));

    const cur = try a.alloc(llama.impl.llama_c.llama_token_data, @intCast(n_logits));
    errdefer a.free(cur);
    if (sampled_ids != null) {
        for (cur, 0..) |*t, i| t.* = .{ .id = sampled_ids.?[i], .logit = logits[i], .p = 0.0 };
    } else {
        for (cur, 0..) |*t, i| t.* = .{ .id = @intCast(i), .logit = logits[i], .p = 0.0 };
    }

    // sort tokens by logits (partial: only the leading `n_top` need ordering)
    if (n_top > cur.len) {
        n_top = cur.len;
    }
    if (n_top > 0) {
        stdlib.libcxx.partialSort(llama.impl.llama_c.llama_token_data, cur, n_top, {}, logitGreater);
    }

    // apply softmax
    var max_l = -std.math.inf(f32);
    if (n_top > 0) {
        max_l = cur[0].logit; // partial_sort guarantees the absolute maximum is at index 0
    } else {
        for (cur) |t| {
            max_l = @max(max_l, t.logit);
        }
    }
    var cum_sum: f32 = 0.0;
    for (cur) |*t| {
        const p = stdlib.libm.expf(t.logit - max_l);
        t.p = p;
        cum_sum += p;
    }
    for (cur) |*t| {
        t.p /= cum_sum;
    }

    return cur;
}

/// Ports `safe_json_to_str` (server-common.cpp:1785 @b1b770d90): compact, with
/// bad UTF-8 replaced instead of throwing.
///
/// Return: owned by `a`.
pub fn safeJsonToStr(a: std.mem.Allocator, data: *const json.Value) error{OutOfMemory}![]u8 {
    return json.dumpSafe(a, data, -1);
}

/// Ports the `tokens_to_str` overload over a vocabulary (server-common.cpp:1805
/// @b1b770d90), through the iterator template (server-common.cpp:1791
/// @b1b770d90): every token's piece, concatenated.
///
/// Return: owned by `a`.
pub fn tokensToStr(a: std.mem.Allocator, vocab: ?*const llama.impl.llama_c.llama_vocab, tokens: []const llama.impl.llama_c.llama_token) common_chat.exception.Error![]u8 {
    var ret: std.ArrayList(u8) = .empty;
    errdefer ret.deinit(a);
    for (tokens) |t| {
        const pc = try piece(a, vocab, t);
        defer a.free(pc);
        try ret.appendSlice(a, pc);
    }
    return ret.toOwnedSlice(a);
}

/// Ports the `tokens_to_str` overload over a context (server-common.cpp:1800 @b1b770d90).
pub fn tokensToStrCtx(a: std.mem.Allocator, ctx: ?*llama.impl.llama_c.llama_context, tokens: []const llama.impl.llama_c.llama_token) common_chat.exception.Error![]u8 {
    const model = llama.api.context.llama_get_model(ctx);
    return tokensToStr(a, llama.api.model.llama_model_get_vocab(model), tokens);
}

/// Ports `common_tokenize` (common/common.cpp:1866 @b1b770d90).
///
/// Return: owned by `a`.
pub fn tokenize(a: std.mem.Allocator, vocab: ?*const llama.impl.llama_c.llama_vocab, text: []const u8, add_special: bool, parse_special: bool) common_chat.exception.Error![]llama.impl.llama_c.llama_token {
    // upper limit for the number of tokens
    var n_tokens: c_int = @truncate(@as(i64, @intCast(text.len)) + 2 * @as(i64, @intFromBool(add_special)));
    var result = try a.alloc(llama.impl.llama_c.llama_token, @intCast(@max(n_tokens, 0)));
    errdefer a.free(result);
    // libllama's throw -- a vocabulary with no byte token for a byte of the
    // text -- reaches the caller as an exception, which the handlers'
    // `ex_wrapper` answers with 500, as upstream's does. `llama_tokenize`
    // itself would terminate: an exception cannot leave its C ABI.
    var exc: llama.impl.Exception = .{};
    n_tokens = llama.api.vocab.tokenize(@ptrCast(vocab), text, result, add_special, parse_special, &exc) catch
        return common_chat.exception.throw(error.RuntimeError, "{s}", .{exc.what()});
    if (n_tokens == std.math.minInt(i32)) {
        return common_chat.exception.throw(error.RuntimeError, "Tokenization failed: input text too large, tokenization result exceeds int32_t limit", .{});
    }
    if (n_tokens < 0) {
        result = try a.realloc(result, @intCast(-n_tokens));
        const check = llama.api.vocab.tokenize(@ptrCast(vocab), text, result, add_special, parse_special, &exc) catch
            return common_chat.exception.throw(error.RuntimeError, "{s}", .{exc.what()});
        tools_common.ggml.ggmlAssert(check == -n_tokens, "common.cpp", 1881, "check == -n_tokens");
    } else {
        result = try a.realloc(result, @intCast(n_tokens));
    }
    return result;
}

/// Ports `common_detokenize` (common/common.cpp:1916 @b1b770d90), the overload
/// over a vocabulary.
///
/// Return: owned by `a`.
pub fn detokenize(a: std.mem.Allocator, vocab: ?*const llama.impl.llama_c.llama_vocab, tokens: []const llama.impl.llama_c.llama_token, special: bool) common_chat.exception.Error![]u8 {
    var text = try a.alloc(u8, @max(22, tokens.len));
    errdefer a.free(text);
    // A token outside the vocabulary throws in libllama, as in `tokenize`.
    var exc: llama.impl.Exception = .{};
    var n_chars = llama.api.vocab.detokenize(@ptrCast(vocab), tokens, text, false, special, &exc) catch
        return common_chat.exception.throw(error.RuntimeError, "{s}", .{exc.what()});
    if (n_chars < 0) {
        text = try a.realloc(text, @intCast(-n_chars));
        n_chars = llama.api.vocab.detokenize(@ptrCast(vocab), tokens, text, false, special, &exc) catch
            return common_chat.exception.throw(error.RuntimeError, "{s}", .{exc.what()});
        tools_common.ggml.ggmlAssert(n_chars <= @as(i32, @intCast(text.len)), "common.cpp", 1923, "n_chars <= (int32_t)text.size()"); // whitespace trimming is performed after per-token detokenization
    }

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return a.realloc(text, @intCast(n_chars));
}

/// Ports `common_token_to_piece` (common/common.cpp:1894 @b1b770d90), the
/// overload over a vocabulary, with `special = true`, the default.
pub fn piece(a: std.mem.Allocator, vocab: ?*const llama.impl.llama_c.llama_vocab, t: llama.impl.llama_c.llama_token) common_chat.exception.Error![]u8 {
    // the first try fits libc++'s short-string buffer
    var buf: [22]u8 = undefined;
    // A token outside the vocabulary -- `/detokenize` takes them from the
    // client -- throws in libllama, as in `tokenize`.
    var exc: llama.impl.Exception = .{};
    const n_chars = llama.api.vocab.tokenToPiece(@ptrCast(vocab), t, &buf, 0, true, &exc) catch
        return common_chat.exception.throw(error.RuntimeError, "{s}", .{exc.what()});
    if (n_chars < 0) {
        const out = try a.alloc(u8, @intCast(-n_chars));
        errdefer a.free(out);
        const check = llama.api.vocab.tokenToPiece(@ptrCast(vocab), t, out, 0, true, &exc) catch
            return common_chat.exception.throw(error.RuntimeError, "{s}", .{exc.what()});
        tools_common.ggml.ggmlAssert(check == -n_chars, "common.cpp", 1901, "check == -n_chars");
        return out;
    }
    return a.dupe(u8, buf[0..@intCast(n_chars)]);
}

/// Ports `tokens_to_output_formatted_string` (server-common.cpp:1810 @b1b770d90):
/// the token's piece, or `byte: \xNN` for a lone byte of a multi-byte
/// character.
///
/// Return: owned by `a`.
pub fn tokensToOutputFormattedString(a: std.mem.Allocator, ctx: ?*const llama.impl.llama_c.llama_context, token: llama.impl.llama_c.llama_token) common_chat.exception.Error![]u8 {
    var out: []u8 = if (token == token_null) try a.alloc(u8, 0) else blk: {
        const model = llama.api.context.llama_get_model(ctx);
        break :blk try piece(a, llama.api.model.llama_model_get_vocab(model), token);
    };

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.len == 1 and (out[0] & 0x80) == 0x80) {
        const b = out[0];
        a.free(out);
        out = try std.fmt.allocPrint(a, "byte: \\x{x}", .{b});
    }

    return out;
}

// -----------------------------------------------------------------------------
// SSE

/// Ports `format_oai_sse` (server-common.cpp:1827 @b1b770d90): one `data:`
/// event per array element, or one for the value.
///
/// Return: owned by `a`.
pub fn formatOaiSse(a: std.mem.Allocator, data: *const json.Value) common_chat.exception.Error![]u8 {
    var ss: std.ArrayList(u8) = .empty;
    errdefer ss.deinit(a);
    if (data.isArray()) {
        for (data.array.items) |*item| try sendSingle(a, &ss, item);
    } else {
        try sendSingle(a, &ss, data);
    }
    return ss.toOwnedSlice(a);
}

fn sendSingle(a: std.mem.Allocator, ss: *std.ArrayList(u8), data: *const json.Value) common_chat.exception.Error!void {
    const s = try safeJsonToStr(a, data);
    defer a.free(s);
    try ss.appendSlice(a, "data: ");
    try ss.appendSlice(a, s);
    try ss.appendSlice(a, "\n\n"); // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
}

/// Ports `format_oai_resp_sse` (server-common.cpp:1846 @b1b770d90): `event:`
/// and `data:` lines from each `{event, data}`.
///
/// Return: owned by `a`.
pub fn formatOaiRespSse(a: std.mem.Allocator, data: *const json.Value) common_chat.exception.Error![]u8 {
    var ss: std.ArrayList(u8) = .empty;
    errdefer ss.deinit(a);
    if (data.isArray()) {
        for (data.array.items) |*item| try sendEvent(a, &ss, item);
    } else {
        try sendEvent(a, &ss, data);
    }
    return ss.toOwnedSlice(a);
}

fn sendEvent(a: std.mem.Allocator, ss: *std.ArrayList(u8), event_obj: *const json.Value) common_chat.exception.Error!void {
    try ss.appendSlice(a, "event: ");
    try ss.appendSlice(a, try (try event_obj.at("event")).get([]const u8));
    try ss.appendSlice(a, "\n");
    const s = try safeJsonToStr(a, try event_obj.at("data"));
    defer a.free(s);
    try ss.appendSlice(a, "data: ");
    try ss.appendSlice(a, s);
    try ss.appendSlice(a, "\n\n");
}

/// Ports `format_anthropic_sse` (server-common.cpp:1864 @b1b770d90): an
/// `event:` line where the item has both `event` and `data`, a bare `data:`
/// otherwise.
///
/// Return: owned by `a`.
pub fn formatAnthropicSse(a: std.mem.Allocator, data: *const json.Value) common_chat.exception.Error![]u8 {
    var ss: std.ArrayList(u8) = .empty;
    errdefer ss.deinit(a);
    if (data.isArray()) {
        for (data.array.items) |*event| try sendAnthropic(a, &ss, event);
    } else {
        try sendAnthropic(a, &ss, data);
    }
    return ss.toOwnedSlice(a);
}

fn sendAnthropic(a: std.mem.Allocator, ss: *std.ArrayList(u8), event_obj: *const json.Value) common_chat.exception.Error!void {
    if (event_obj.contains("event") and event_obj.contains("data")) {
        try sendEvent(a, ss, event_obj);
    } else {
        try sendSingle(a, ss, event_obj);
    }
}

// -----------------------------------------------------------------------------
// server_pipe

/// Ports `server_pipe` (server-common.h:633 @b1b770d90): a queue between two
/// threads, with a close flag for each side and an optional bound.
pub fn Pipe(comptime T: type) type {
    return struct {
        const Self = @This();

        mutex: stdlib.sync.Mutex = .{},
        cv: stdlib.sync.Condition = .{},
        queue: std.ArrayList(T) = .empty,
        head: usize = 0,
        writer_closed: std.atomic.Value(bool) = .init(false),
        reader_closed: std.atomic.Value(bool) = .init(false),
        gpa: std.mem.Allocator,

        // 0 = unbounded (default)
        // > 0, write() drops the oldest item once the queue is full
        max_size: usize = 0,

        /// `should_stop`: a context and a predicate; null means "never stop".
        pub const ShouldStop = struct {
            ctx: *anyopaque,
            f: *const fn (*anyopaque) bool,
        };

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa };
        }

        pub fn deinit(self: *Self) void {
            self.queue.deinit(self.gpa);
        }

        fn len(self: *const Self) usize {
            return self.queue.items.len - self.head;
        }

        fn pop(self: *Self) T {
            const v = self.queue.items[self.head];
            self.head += 1;
            if (self.head == self.queue.items.len) {
                self.queue.clearRetainingCapacity();
                self.head = 0;
            }
            return v;
        }

        /// Ports `close_write` (server-common.h:644 @b1b770d90).
        pub fn closeWrite(self: *Self) void {
            self.writer_closed.store(true, .monotonic);
            self.cv.broadcast();
        }

        /// Ports `close_read` (server-common.h:649 @b1b770d90).
        pub fn closeRead(self: *Self) void {
            self.reader_closed.store(true, .monotonic);
            self.cv.broadcast();
        }

        /// Ports `read` (server-common.h:656 @b1b770d90). `close_on_stop`:
        /// should_stop means the reader is gone for good, so the writer is
        /// told the pipe is broken; otherwise it is a per-read deadline.
        ///
        /// Return: true with `output` set; false at a clean EOF or a stop.
        pub fn read(self: *Self, output: *T, should_stop: ?ShouldStop, close_on_stop: bool) bool {
            self.mutex.lock();
            defer self.mutex.unlock();
            const poll_interval_ns: u64 = 500 * std.time.ns_per_ms;
            while (true) {
                if (self.len() != 0) {
                    output.* = self.pop();
                    return true;
                }
                if (self.writer_closed.load(.seq_cst)) {
                    return false; // clean EOF
                }
                if (should_stop) |s| { // a null should_stop means "never stop"
                    if (s.f(s.ctx)) {
                        if (close_on_stop) {
                            self.closeRead(); // signal broken pipe to writer
                        }
                        return false; // cancelled / deadline reached
                    }
                }
                _ = self.cv.timedWait(&self.mutex, poll_interval_ns);
            }
        }

        /// Ports `write` (server-common.h:678 @b1b770d90).
        ///
        /// Return: false when the reader is gone (a broken pipe).
        pub fn write(self: *Self, data: T) common_chat.exception.Error!bool {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.reader_closed.load(.seq_cst)) {
                return false; // broken pipe
            }
            if (self.max_size > 0) {
                while (self.len() >= self.max_size) {
                    _ = self.pop(); // drop oldest to stay bounded
                }
            }
            try self.queue.append(self.gpa, data);
            self.cv.signal();
            return true;
        }
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "validate_utf8 cuts a trailing partial character" {
    try std.testing.expectEqual(@as(usize, 0), validateUtf8(""));
    try std.testing.expectEqual(@as(usize, 1), validateUtf8("a\xc3"));
    try std.testing.expectEqual(@as(usize, 3), validateUtf8("a\xc3\xa9"));
    try std.testing.expectEqual(@as(usize, 1), validateUtf8("a\xe2\x82"));
    try std.testing.expect(isValidUtf8("h\xc3\xa9"));
    try std.testing.expect(!isValidUtf8("\xff"));
}

test "random_string is 32 characters of the alphabet" {
    const a = std.testing.allocator;
    const s = try randomString(a);
    defer a.free(s);
    try std.testing.expectEqual(@as(usize, 32), s.len);
    for (s) |ch| try std.testing.expect(std.ascii.isAlphanumeric(ch));
}

test "a bounded pipe drops the oldest" {
    var p = Pipe(i32).init(std.testing.allocator);
    defer p.deinit();
    p.max_size = 2;
    _ = try p.write(1);
    _ = try p.write(2);
    _ = try p.write(3);
    var v: i32 = 0;
    try std.testing.expect(p.read(&v, null, true));
    try std.testing.expectEqual(@as(i32, 2), v);
    p.closeWrite();
    try std.testing.expect(p.read(&v, null, true));
    try std.testing.expectEqual(@as(i32, 3), v);
    try std.testing.expect(!p.read(&v, null, true));
}
