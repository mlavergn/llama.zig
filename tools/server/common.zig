//! The server's shared helpers: `json_value`, error responses, random ids,
//! the media marker, LoRA bookkeeping, UTF-8 checks, slot statistics and
//! server metrics, token probabilities, the SSE formatters, and the pipe
//! between threads.
//!
//! # Provenance
//!
//! Ports parts of `llama.cpp/tools/server/server-common.cpp` and
//! `llama.cpp/tools/server/server-common.h` (llama.cpp v0.8.0,
//! `8642541b2`). The rest of that pair is `tokens.zig` (`server_tokens` and
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
const json = @import("json");
const chat = @import("common_chat");
const tools = @import("tools_common");
const libcxx = @import("llama").libcxx;
const llama_api = @import("llama").api;
const LlamaException = @import("llama").impl.Exception;

pub const c = tools.c;
const log = tools.log;
pub const exception = chat.exception;

pub const Value = json.Value;
pub const Allocator = std.mem.Allocator;

/// Every throwing call in the server's data layer returns one of these.
pub const Error = exception.Error;

/// `e.what()` for an error the data layer returned on this thread.
pub const what = exception.what;

/// `throw std::runtime_error(...)` / `throw std::invalid_argument(...)`.
pub const throw = exception.throw;

/// `llama_token`.
pub const Token = c.llama_token;

/// `LLAMA_TOKEN_NULL`.
pub const token_null: Token = c.LLAMA_TOKEN_NULL;

// -----------------------------------------------------------------------------
// logging and assertions

/// `GGML_ASSERT(cond)` as a header or source file of the server states it.
pub fn ggmlAssert(ok: bool, comptime file: [*:0]const u8, comptime line: c_int, comptime cond: [*:0]const u8) void {
    if (!ok) _ = c.ggml_abort(file, line, "GGML_ASSERT(%s) failed", cond);
}

/// The levels of the `SRV_*` macros.
pub const SrvLevel = enum { dbg, trc, inf, wrn, err };

/// Ports `SRV_DBG`, `SRV_TRC`, `SRV_INF`, `SRV_WRN` and `SRV_ERR`
/// (server-common.h:35, 36, 37, 38, 39 @8642541b2): the message
/// prefixed `srv  %12.*s: ` with the calling function's name.
pub fn srv(comptime level: SrvLevel, comptime func: []const u8, comptime fmt: []const u8, args: anytype) void {
    const f = "srv  %12.*s: " ++ fmt;
    const pre = .{ @as(c_int, 12), @as([*:0]const u8, func ++ "") };
    switch (level) {
        .dbg => log.dbg(f, pre ++ args),
        .trc => log.trc(f, pre ++ args),
        .inf => log.inf(f, pre ++ args),
        .wrn => log.wrn(f, pre ++ args),
        .err => log.err(f, pre ++ args),
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

/// Ports `json_value` (server-common.h:45 @8642541b2) for a scalar `T`:
/// the member `key` as `T`, or `default_value` when it is absent or null,
/// or -- with a warning -- when it does not convert.
///
/// Parameters:
/// - `body`: any value; only an object has members.
/// - `T`: `bool`, an integer or float type, or `[]const u8` (borrowed from
///   `body`).
///
/// Return: the value or the default.
pub fn jsonValue(body: *const Value, key: []const u8, comptime T: type, default_value: T) T {
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
pub fn jsonValueJson(gpa: Allocator, body: *const Value, key: []const u8, default_value: Value) Error!Value {
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
pub fn jsonValueStrings(gpa: Allocator, body: *const Value, key: []const u8) Error![]const []const u8 {
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
    log.wrn("Wrong type supplied for parameter '%s', using default value: %s\n", .{ k, w });
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
pub fn from(a: Allocator, x: anytype) Error!Value {
    const T = @TypeOf(x);
    if (T == Value) return x;
    if (T == Bytes) {
        var arr = Value.emptyArray();
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
            if (Elem == u8) return Value.from(a, x);
            var arr = Value.emptyArray();
            errdefer arr.deinit(a);
            for (x) |e| try arr.pushBack(a, try from(a, e));
            return arr;
        },
        else => return Value.from(a, x),
    }
}

/// A JSON object under construction, members in insertion order.
pub const Obj = struct {
    a: Allocator,
    v: Value = Value.emptyObject(),

    pub fn init(a: Allocator) Obj {
        return .{ .a = a };
    }

    /// `obj[key] = x`.
    pub fn put(self: *Obj, key: []const u8, x: anytype) Error!void {
        try self.v.set(self.a, key, try from(self.a, x));
    }

    /// Return: the object, owned by the caller.
    pub fn done(self: *Obj) Value {
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

/// Ports `error_type` (server-common.h:60 @8642541b2).
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

/// Ports `format_error_response` (server-common.cpp:93 @8642541b2).
///
/// Return: `{"code", "message", "type"}`, owned by the caller.
pub fn formatErrorResponse(a: Allocator, message: []const u8, ty: ErrorType) Error!Value {
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

/// Ports `common_grammar_trigger_type` (common/common.h:148 @8642541b2). The
/// server casts any integer to it, so it is kept as one.
pub const trigger_type_token: c_int = 0;
pub const trigger_type_word: c_int = 1;
pub const trigger_type_pattern: c_int = 2;
pub const trigger_type_pattern_full: c_int = 3;

/// Ports `common_grammar_trigger` (common/common.h:155 @8642541b2), as
/// `server_grammar_trigger` (server-common.h:72 @8642541b2) wraps it.
pub const GrammarTrigger = struct {
    type: c_int = 0,
    value: []const u8 = "",
    token: Token = token_null,

    /// Ports `server_grammar_trigger`'s constructor from JSON
    /// (server-common.h:72 @8642541b2).
    ///
    /// Return: `value` is `a`'s.
    pub fn fromJson(a: Allocator, in: *const Value) Error!GrammarTrigger {
        var t: GrammarTrigger = .{};
        t.type = try (try in.at("type")).get(c_int);
        t.value = try a.dupe(u8, try (try in.at("value")).get([]const u8));
        if (t.type == trigger_type_token) {
            t.token = try (try in.at("token")).get(c_int);
        }
        return t;
    }

    /// Ports `server_grammar_trigger`'s `to_json` (server-common.h:85 @8642541b2).
    ///
    /// Return: owned by the caller.
    pub fn toJson(self: GrammarTrigger, a: Allocator) Error!Value {
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

extern fn arc4random() u32;

/// One `std::random_device` draw: libc++ reads `arc4random` on Darwin and
/// `getrandom` elsewhere; either is an unpredictable 32-bit value.
pub fn randomDevice() u32 {
    if (comptime @import("builtin").os.tag.isDarwin()) return arc4random();
    var b: [4]u8 = undefined;
    _ = std.os.linux.getrandom(&b, b.len, 0);
    return std.mem.readInt(u32, &b, .little);
}

/// Ports `random_string` (server-common.cpp:168 @8642541b2): 32 characters of
/// `[0-9A-Za-z]` from an `mt19937` seeded by the random device.
///
/// Return: owned by the caller.
pub fn randomString(a: Allocator) Error![]u8 {
    const str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    var generator = libcxx.Mt19937.init(randomDevice());
    const result = try a.alloc(u8, 32);
    for (result) |*ch| ch.* = str[generator.next() % str.len];
    return result;
}

/// Ports `gen_chatcmplid` (server-common.cpp:183 @8642541b2).
pub fn genChatcmplid(a: Allocator) Error![]u8 {
    const r = try randomString(a);
    defer a.free(r);
    return std.mem.concat(a, u8, &.{ "chatcmpl-", r });
}

/// Ports `gen_tool_call_id` (server-common.cpp:187 @8642541b2).
pub fn genToolCallId(a: Allocator) Error![]u8 {
    return randomString(a);
}

var media_marker_mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var media_marker_done = false;
var media_marker_buf: [64]u8 = undefined;
var media_marker: [:0]const u8 = "";

fn initMediaMarker() void {
    // allow user to pin a reproducible marker via env var
    if (std.c.getenv("LLAMA_MEDIA_MARKER")) |env| {
        if (env[0] != 0) {
            media_marker = std.heap.c_allocator.dupeSentinel(u8, std.mem.span(env), 0) catch "";
            return;
        }
    }
    var r: [32]u8 = undefined;
    const s = randomString(std.heap.c_allocator) catch return;
    @memcpy(&r, s);
    std.heap.c_allocator.free(s);
    media_marker = std.fmt.bufPrintSentinel(&media_marker_buf, "<__media_{s}__>", .{r}, 0) catch "";
}

/// Ports `get_media_marker` (server-common.cpp:191 @8642541b2): fixed for the
/// process's life, from `LLAMA_MEDIA_MARKER` when set.
pub fn getMediaMarker() [:0]const u8 {
    // a function-local static: initialized once, thread-safe
    _ = std.c.pthread_mutex_lock(&media_marker_mutex);
    defer _ = std.c.pthread_mutex_unlock(&media_marker_mutex);
    if (!media_marker_done) {
        initMediaMarker();
        media_marker_done = true;
    }
    return media_marker;
}

// -----------------------------------------------------------------------------
// model output modalities

/// Ports `server_model_output_modalities` (server-common.cpp:207
/// @8642541b2): output modalities for `architecture.output_modalities` in
/// GET /models.
///
/// Parameters:
/// - `has_cap_chat`: false for a tts-only model.
/// - `has_cap_tts`: the model can generate audio.
///
/// Return: static strings.
pub fn serverModelOutputModalities(decision_type: tools.common.DecisionType, has_cap_chat: bool, has_cap_tts: bool) []const []const u8 {
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

/// Ports `server_model_architecture_json` (server-common.cpp:248
/// @8642541b2): the architecture object of GET /models; shared by the
/// direct server and the router.
///
/// Return: owned by `a`.
pub fn serverModelArchitectureJson(a: Allocator, inp_image: bool, inp_audio: bool, inp_video: bool, output_modalities: []const []const u8) Error!Value {
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

/// `common_adapter_lora_info`, as the server's LoRA helpers read it.
pub const LoraInfo = tools.params.LoraInfo;

/// Ports `lora_all_alora` (server-common.cpp:274 @8642541b2).
pub fn loraAllAlora(loras: []const LoraInfo) bool {
    var found_alora = false;
    for (loras) |lora| {
        if (lora.scale != 0) {
            if (c.llama_adapter_get_alora_n_invocation_tokens(lora.ptr) == 0) {
                return false;
            }
            found_alora = true;
        }
    }
    return found_alora;
}

/// Ports `lora_should_clear_cache` (server-common.cpp:287 @8642541b2).
pub fn loraShouldClearCache(current: []const LoraInfo, next: []const LoraInfo) bool {
    // This should always be called after determining that the two sets are
    // _not_ equal. This assert is therefore some slightly wasted work and
    // should be safe to remove as long as this method is called correctly.
    ggmlAssert(!areLoraEqual(current, next), "server-common.cpp", 294, "!are_lora_equal(current, next)");

    return (!(loraEnabledEmpty(current) or loraAllAlora(current)) or !loraAllAlora(next));
}

fn loraEnabledEmpty(loras: []const LoraInfo) bool {
    for (loras) |l| if (l.scale > 0) return false;
    return true;
}

/// One entry of `std::map<int, float>`, which sorts by id.
pub const LoraScale = struct { id: c_int, scale: f32 };

/// `std::map<int, float>`: an adapter's id to its scale, sorted by id.
pub const LoraMap = struct {
    entries: std.ArrayList(LoraScale) = .empty,

    /// `lora[id] = scale`.
    pub fn put(self: *LoraMap, a: Allocator, id: c_int, scale: f32) Error!void {
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

    pub fn deinit(self: *LoraMap, a: Allocator) void {
        self.entries.deinit(a);
    }

    /// A copy owned by `a`.
    pub fn clone(self: *const LoraMap, a: Allocator) Error!LoraMap {
        return .{ .entries = .fromOwnedSlice(try a.dupe(LoraScale, self.entries.items)) };
    }
};

/// Ports `parse_lora_request` (server-common.cpp:301 @8642541b2).
///
/// Return: owned by `a`.
pub fn parseLoraRequest(a: Allocator, data: *Value) Error!LoraMap {
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

/// Ports `are_lora_equal` (server-common.cpp:314 @8642541b2).
pub fn areLoraEqual(l1: []const LoraInfo, l2: []const LoraInfo) bool {
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

/// Ports `lora_get_enabled_ids` (server-common.cpp:329 @8642541b2).
///
/// Return: owned by `a`.
pub fn loraGetEnabledIds(a: Allocator, loras: []const LoraInfo) Error![]usize {
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

/// Ports `json_is_array_of_numbers` (server-common.cpp:931 @8642541b2).
pub fn jsonIsArrayOfNumbers(data: *const Value) bool {
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

/// Ports `json_is_array_of_mixed_numbers_strings` (server-common.cpp:943 @8642541b2).
pub fn jsonIsArrayOfMixedNumbersStrings(data: *const Value) bool {
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

/// Ports `json_is_array_and_contains_numbers` (server-common.cpp:958 @8642541b2).
pub fn jsonIsArrayAndContainsNumbers(data: *const Value) bool {
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

/// Ports `json_get_nested_values` (server-common.cpp:970 @8642541b2): each
/// path's value, keyed by the path, when every key along it exists.
///
/// Return: owned by `a`.
pub fn jsonGetNestedValues(a: Allocator, paths: []const []const u8, js: *const Value) Error!Value {
    var result = Value.emptyObject();
    errdefer result.deinit(a);

    for (paths) |path| {
        var current: *const Value = js;
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

/// Ports `validate_utf8` (server-common.cpp:1027 @8642541b2): the length of
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

/// Ports `is_valid_utf8` (server-common.cpp:1818 @8642541b2): lead bytes and
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

/// Ports `server_slot_stats` (server-common.h:406 @8642541b2): shared between
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

    /// Ports `update_prompt_start` (server-common.h:424 @8642541b2).
    pub fn updatePromptStart(self: *SlotStats) void {
        ggmlAssert(self.t_start == 0, "server-common.h", 425, "t_start == 0");
        self.t_start = c.ggml_time_us();
    }

    /// Ports `set_prompt_last` (server-common.h:428 @8642541b2).
    pub fn setPromptLast(self: *SlotStats, t_us: i64) void {
        ggmlAssert(self.t_start > 0, "server-common.h", 429, "t_start > 0");
        self.t_prompt_last = t_us;
    }

    /// Ports `update_prompt_last` (server-common.h:432 @8642541b2).
    pub fn updatePromptLast(self: *SlotStats) void {
        self.setPromptLast(c.ggml_time_us());
    }

    /// Ports `update_gen_last` (server-common.h:435 @8642541b2).
    pub fn updateGenLast(self: *SlotStats) void {
        ggmlAssert(self.t_prompt_last > 0, "server-common.h", 436, "t_prompt_last > 0");
        self.t_gen_last = c.ggml_time_us();
    }

    /// Ports `t_elapsed_us` (server-common.h:441 @8642541b2).
    pub fn tElapsedUs(self: *const SlotStats) i64 {
        return c.ggml_time_us() - self.t_start;
    }

    /// Ports `t_prompt_ms` (server-common.h:444 @8642541b2).
    pub fn tPromptMs(self: *const SlotStats) f64 {
        if (self.t_prompt_last == 0) {
            return 0.0; // the prompt is not processed yet
        }
        return @as(f64, @floatFromInt(self.t_prompt_last - self.t_start)) / 1000.0;
    }

    /// Ports `t_gen_us` (server-common.h:450 @8642541b2).
    pub fn tGenUs(self: *const SlotStats) i64 {
        if (self.t_gen_last == 0) {
            return 0; // the generation is not started yet
        }
        // clamp to 1 us, the first token can land in the same us as t_prompt_last
        return @max(1, self.t_gen_last - self.t_prompt_last);
    }

    /// Ports `t_gen_ms` (server-common.h:457 @8642541b2).
    pub fn tGenMs(self: *const SlotStats) f64 {
        return @as(f64, @floatFromInt(self.tGenUs())) / 1000.0;
    }

    /// Ports `n_gen_steps` (server-common.h:463 @8642541b2).
    pub fn nGenSteps(self: *const SlotStats) u64 {
        return if (self.n_gen > 0) self.n_gen - 1 else 0;
    }

    /// Ports `t_prompt_per_token_ms` (server-common.h:469 @8642541b2).
    pub fn tPromptPerTokenMs(self: *const SlotStats) f64 {
        return if (self.n_prompt_processed > 0) self.tPromptMs() / @as(f64, @floatFromInt(self.n_prompt_processed)) else 0.0;
    }

    /// Ports `t_gen_per_token_ms` (server-common.h:472 @8642541b2).
    pub fn tGenPerTokenMs(self: *const SlotStats) f64 {
        return if (self.nGenSteps() > 0) self.tGenMs() / @as(f64, @floatFromInt(self.nGenSteps())) else 0.0;
    }

    /// Ports `n_prompt_tps` (server-common.h:475 @8642541b2).
    pub fn nPromptTps(self: *const SlotStats) f64 {
        const t_ms = self.tPromptMs();
        return if (t_ms > 0.0) 1e3 / t_ms * @as(f64, @floatFromInt(self.n_prompt_processed)) else 0.0;
    }

    /// Ports `n_gen_tps` (server-common.h:479 @8642541b2).
    pub fn nGenTps(self: *const SlotStats) f64 {
        const t_ms = self.tGenMs();
        return if (t_ms > 0.0) 1e3 / t_ms * @as(f64, @floatFromInt(self.nGenSteps())) else 0.0;
    }

    /// Ports `is_set` (server-common.h:485 @8642541b2).
    pub fn isSet(self: *const SlotStats) bool {
        return self.t_start > 0;
    }

    /// Ports `server_slot_stats`'s `to_json` (server-common.cpp:141 @8642541b2).
    ///
    /// Return: owned by `a`.
    pub fn toJson(self: *const SlotStats, a: Allocator) Error!Value {
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

/// Ports `server_metrics` (server-common.h:494 @8642541b2): server-global and
/// cumulative.
pub const Metrics = struct {
    /// Ports `server_metrics`'s `bucket` (server-common.h:497 @8642541b2).
    pub const Bucket = struct {
        count: u64 = 0, // number of tokens
        steps: u64 = 0, // number of decode steps,
        // this excludes first generated token (logits from prompt batch)
        time: u64 = 0, // in microseconds

        /// Ports `n_per_second` (server-common.h:504 @8642541b2): the rate uses
        /// the decode steps, so that "free" tokens do not inflate it.
        pub fn nPerSecond(self: Bucket) f64 {
            return if (self.time > 0) @as(f64, @floatFromInt(self.steps)) / @as(f64, @floatFromInt(self.time)) * 1e6 else 0.0;
        }

        /// Ports `add` (server-common.h:508 @8642541b2).
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

    /// Ports `init` (server-common.h:536 @8642541b2).
    pub fn init(self: *Metrics) void {
        self.t_start = c.ggml_time_us();
    }

    /// Ports `reset_bucket` (server-common.h:540 @8642541b2).
    pub fn resetBucket(self: *Metrics) void {
        self.prompt_bucket = .{};
        self.predict_bucket = .{};
    }

    /// Ports `add_prompt` (server-common.h:545 @8642541b2).
    pub fn addPrompt(self: *Metrics, n_tokens: u64, t_us: u64) void {
        self.prompt.add(n_tokens, n_tokens, t_us);
        self.prompt_bucket.add(n_tokens, n_tokens, t_us);
    }

    /// Ports `add_prompt_cached` (server-common.h:550 @8642541b2).
    pub fn addPromptCached(self: *Metrics, n_tokens: u64) void {
        self.n_prompt_cached +%= n_tokens;
    }

    /// A copy whose list `a` owns, as the C++ copy constructor makes one.
    pub fn clone(self: *const Metrics, a: Allocator) Error!Metrics {
        var m = self.*;
        m.n_accepted_per_pos = .fromOwnedSlice(try a.dupe(u64, self.n_accepted_per_pos.items));
        return m;
    }

    pub fn deinit(self: *Metrics, a: Allocator) void {
        self.n_accepted_per_pos.deinit(a);
    }
};

// -----------------------------------------------------------------------------
// other utils

/// `llama_token_data`.
pub const TokenData = c.llama_token_data;

fn logitGreater(_: void, x: TokenData, y: TokenData) bool {
    return x.logit > y.logit;
}

/// Ports `get_token_probabilities` (server-common.cpp:1664 @8642541b2): the
/// candidates at output `idx`, the leading `n_top` sorted by logit
/// (libc++'s `partial_sort`), with their softmax.
///
/// Return: owned by `a`.
pub fn getTokenProbabilities(a: Allocator, ctx: ?*c.llama_context, idx: c_int, n_top_in: usize) Error![]TokenData {
    var n_top = n_top_in;
    const logits = c.llama_get_logits_ith(ctx, idx);
    const sampled_ids = c.llama_get_sampled_candidates_ith(ctx, idx);

    const n_logits: c_int = @bitCast(c.llama_get_sampled_logits_count_ith(ctx, idx));

    const cur = try a.alloc(TokenData, @intCast(n_logits));
    errdefer a.free(cur);
    if (sampled_ids != null) {
        for (cur, 0..) |*t, i| t.* = .{ .id = sampled_ids[i], .logit = logits[i], .p = 0.0 };
    } else {
        for (cur, 0..) |*t, i| t.* = .{ .id = @intCast(i), .logit = logits[i], .p = 0.0 };
    }

    // sort tokens by logits (partial: only the leading `n_top` need ordering)
    if (n_top > cur.len) {
        n_top = cur.len;
    }
    if (n_top > 0) {
        libcxx.partialSort(TokenData, cur, n_top, {}, logitGreater);
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
        const p = expf(t.logit - max_l);
        t.p = p;
        cum_sum += p;
    }
    for (cur) |*t| {
        t.p /= cum_sum;
    }

    return cur;
}

extern fn expf(x: f32) f32;

/// Ports `safe_json_to_str` (server-common.cpp:1716 @8642541b2): compact, with
/// bad UTF-8 replaced instead of throwing.
///
/// Return: owned by `a`.
pub fn safeJsonToStr(a: Allocator, data: *const Value) error{OutOfMemory}![]u8 {
    return json.dumpSafe(a, data, -1);
}

/// Ports the `tokens_to_str` overload over a vocabulary (server-common.cpp:1736
/// @8642541b2), through the iterator template (server-common.cpp:1722
/// @8642541b2): every token's piece, concatenated.
///
/// Return: owned by `a`.
pub fn tokensToStr(a: Allocator, vocab: ?*const c.llama_vocab, tokens: []const Token) Error![]u8 {
    var ret: std.ArrayList(u8) = .empty;
    errdefer ret.deinit(a);
    for (tokens) |t| {
        const pc = try piece(a, vocab, t);
        defer a.free(pc);
        try ret.appendSlice(a, pc);
    }
    return ret.toOwnedSlice(a);
}

/// Ports the `tokens_to_str` overload over a context (server-common.cpp:1731 @8642541b2).
pub fn tokensToStrCtx(a: Allocator, ctx: ?*c.llama_context, tokens: []const Token) Error![]u8 {
    const model = c.llama_get_model(ctx);
    return tokensToStr(a, c.llama_model_get_vocab(model), tokens);
}

/// Ports `common_tokenize` (common/common.cpp:1866 @8642541b2).
///
/// Return: owned by `a`.
pub fn tokenize(a: Allocator, vocab: ?*const c.llama_vocab, text: []const u8, add_special: bool, parse_special: bool) Error![]Token {
    // upper limit for the number of tokens
    var n_tokens: c_int = @truncate(@as(i64, @intCast(text.len)) + 2 * @as(i64, @intFromBool(add_special)));
    var result = try a.alloc(Token, @intCast(@max(n_tokens, 0)));
    errdefer a.free(result);
    // libllama's throw -- a vocabulary with no byte token for a byte of the
    // text -- reaches the caller as an exception, which the handlers'
    // `ex_wrapper` answers with 500, as upstream's does. `llama_tokenize`
    // itself would terminate: an exception cannot leave its C ABI.
    var exc: LlamaException = .{};
    n_tokens = llama_api.vocab.tokenize(@ptrCast(vocab), text, result, add_special, parse_special, &exc) catch
        return throw(error.RuntimeError, "{s}", .{exc.what()});
    if (n_tokens == std.math.minInt(i32)) {
        return throw(error.RuntimeError, "Tokenization failed: input text too large, tokenization result exceeds int32_t limit", .{});
    }
    if (n_tokens < 0) {
        result = try a.realloc(result, @intCast(-n_tokens));
        const check = llama_api.vocab.tokenize(@ptrCast(vocab), text, result, add_special, parse_special, &exc) catch
            return throw(error.RuntimeError, "{s}", .{exc.what()});
        ggmlAssert(check == -n_tokens, "common.cpp", 1881, "check == -n_tokens");
    } else {
        result = try a.realloc(result, @intCast(n_tokens));
    }
    return result;
}

/// Ports `common_detokenize` (common/common.cpp:1916 @8642541b2), the overload
/// over a vocabulary.
///
/// Return: owned by `a`.
pub fn detokenize(a: Allocator, vocab: ?*const c.llama_vocab, tokens: []const Token, special: bool) Error![]u8 {
    var text = try a.alloc(u8, @max(22, tokens.len));
    errdefer a.free(text);
    // A token outside the vocabulary throws in libllama, as in `tokenize`.
    var exc: LlamaException = .{};
    var n_chars = llama_api.vocab.detokenize(@ptrCast(vocab), tokens, text, false, special, &exc) catch
        return throw(error.RuntimeError, "{s}", .{exc.what()});
    if (n_chars < 0) {
        text = try a.realloc(text, @intCast(-n_chars));
        n_chars = llama_api.vocab.detokenize(@ptrCast(vocab), tokens, text, false, special, &exc) catch
            return throw(error.RuntimeError, "{s}", .{exc.what()});
        ggmlAssert(n_chars <= @as(i32, @intCast(text.len)), "common.cpp", 1923, "n_chars <= (int32_t)text.size()"); // whitespace trimming is performed after per-token detokenization
    }

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return a.realloc(text, @intCast(n_chars));
}

/// Ports `common_token_to_piece` (common/common.cpp:1894 @8642541b2), the
/// overload over a vocabulary, with `special = true`, the default.
pub fn piece(a: Allocator, vocab: ?*const c.llama_vocab, t: Token) Error![]u8 {
    // the first try fits libc++'s short-string buffer
    var buf: [22]u8 = undefined;
    // A token outside the vocabulary -- `/detokenize` takes them from the
    // client -- throws in libllama, as in `tokenize`.
    var exc: LlamaException = .{};
    const n_chars = llama_api.vocab.tokenToPiece(@ptrCast(vocab), t, &buf, 0, true, &exc) catch
        return throw(error.RuntimeError, "{s}", .{exc.what()});
    if (n_chars < 0) {
        const out = try a.alloc(u8, @intCast(-n_chars));
        errdefer a.free(out);
        const check = llama_api.vocab.tokenToPiece(@ptrCast(vocab), t, out, 0, true, &exc) catch
            return throw(error.RuntimeError, "{s}", .{exc.what()});
        ggmlAssert(check == -n_chars, "common.cpp", 1901, "check == -n_chars");
        return out;
    }
    return a.dupe(u8, buf[0..@intCast(n_chars)]);
}

/// Ports `tokens_to_output_formatted_string` (server-common.cpp:1741 @8642541b2):
/// the token's piece, or `byte: \xNN` for a lone byte of a multi-byte
/// character.
///
/// Return: owned by `a`.
pub fn tokensToOutputFormattedString(a: Allocator, ctx: ?*const c.llama_context, token: Token) Error![]u8 {
    var out: []u8 = if (token == token_null) try a.alloc(u8, 0) else blk: {
        const model = c.llama_get_model(ctx);
        break :blk try piece(a, c.llama_model_get_vocab(model), token);
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

/// Ports `format_oai_sse` (server-common.cpp:1758 @8642541b2): one `data:`
/// event per array element, or one for the value.
///
/// Return: owned by `a`.
pub fn formatOaiSse(a: Allocator, data: *const Value) Error![]u8 {
    var ss: std.ArrayList(u8) = .empty;
    errdefer ss.deinit(a);
    if (data.isArray()) {
        for (data.array.items) |*item| try sendSingle(a, &ss, item);
    } else {
        try sendSingle(a, &ss, data);
    }
    return ss.toOwnedSlice(a);
}

fn sendSingle(a: Allocator, ss: *std.ArrayList(u8), data: *const Value) Error!void {
    const s = try safeJsonToStr(a, data);
    defer a.free(s);
    try ss.appendSlice(a, "data: ");
    try ss.appendSlice(a, s);
    try ss.appendSlice(a, "\n\n"); // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
}

/// Ports `format_oai_resp_sse` (server-common.cpp:1777 @8642541b2): `event:`
/// and `data:` lines from each `{event, data}`.
///
/// Return: owned by `a`.
pub fn formatOaiRespSse(a: Allocator, data: *const Value) Error![]u8 {
    var ss: std.ArrayList(u8) = .empty;
    errdefer ss.deinit(a);
    if (data.isArray()) {
        for (data.array.items) |*item| try sendEvent(a, &ss, item);
    } else {
        try sendEvent(a, &ss, data);
    }
    return ss.toOwnedSlice(a);
}

fn sendEvent(a: Allocator, ss: *std.ArrayList(u8), event_obj: *const Value) Error!void {
    try ss.appendSlice(a, "event: ");
    try ss.appendSlice(a, try (try event_obj.at("event")).get([]const u8));
    try ss.appendSlice(a, "\n");
    const s = try safeJsonToStr(a, try event_obj.at("data"));
    defer a.free(s);
    try ss.appendSlice(a, "data: ");
    try ss.appendSlice(a, s);
    try ss.appendSlice(a, "\n\n");
}

/// Ports `format_anthropic_sse` (server-common.cpp:1795 @8642541b2): an
/// `event:` line where the item has both `event` and `data`, a bare `data:`
/// otherwise.
///
/// Return: owned by `a`.
pub fn formatAnthropicSse(a: Allocator, data: *const Value) Error![]u8 {
    var ss: std.ArrayList(u8) = .empty;
    errdefer ss.deinit(a);
    if (data.isArray()) {
        for (data.array.items) |*event| try sendAnthropic(a, &ss, event);
    } else {
        try sendAnthropic(a, &ss, data);
    }
    return ss.toOwnedSlice(a);
}

fn sendAnthropic(a: Allocator, ss: *std.ArrayList(u8), event_obj: *const Value) Error!void {
    if (event_obj.contains("event") and event_obj.contains("data")) {
        try sendEvent(a, ss, event_obj);
    } else {
        try sendSingle(a, ss, event_obj);
    }
}

// -----------------------------------------------------------------------------
// server_pipe

/// Ports `server_pipe` (server-common.h:617 @8642541b2): a queue between two
/// threads, with a close flag for each side and an optional bound.
pub fn Pipe(comptime T: type) type {
    return struct {
        const Self = @This();

        mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
        cv: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,
        queue: std.ArrayList(T) = .empty,
        head: usize = 0,
        writer_closed: std.atomic.Value(bool) = .init(false),
        reader_closed: std.atomic.Value(bool) = .init(false),
        gpa: Allocator,

        // 0 = unbounded (default)
        // > 0, write() drops the oldest item once the queue is full
        max_size: usize = 0,

        /// `should_stop`: a context and a predicate; null means "never stop".
        pub const ShouldStop = struct {
            ctx: *anyopaque,
            f: *const fn (*anyopaque) bool,
        };

        pub fn init(gpa: Allocator) Self {
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

        /// Ports `close_write` (server-common.h:628 @8642541b2).
        pub fn closeWrite(self: *Self) void {
            self.writer_closed.store(true, .monotonic);
            _ = std.c.pthread_cond_broadcast(&self.cv);
        }

        /// Ports `close_read` (server-common.h:633 @8642541b2).
        pub fn closeRead(self: *Self) void {
            self.reader_closed.store(true, .monotonic);
            _ = std.c.pthread_cond_broadcast(&self.cv);
        }

        /// Ports `read` (server-common.h:640 @8642541b2). `close_on_stop`:
        /// should_stop means the reader is gone for good, so the writer is
        /// told the pipe is broken; otherwise it is a per-read deadline.
        ///
        /// Return: true with `output` set; false at a clean EOF or a stop.
        pub fn read(self: *Self, output: *T, should_stop: ?ShouldStop, close_on_stop: bool) bool {
            _ = std.c.pthread_mutex_lock(&self.mutex);
            defer _ = std.c.pthread_mutex_unlock(&self.mutex);
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
                var ts: std.c.timespec = undefined;
                _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
                const ns = @as(u64, @intCast(ts.nsec)) + poll_interval_ns;
                ts.sec += @intCast(ns / std.time.ns_per_s);
                ts.nsec = @intCast(ns % std.time.ns_per_s);
                _ = std.c.pthread_cond_timedwait(&self.cv, &self.mutex, &ts);
            }
        }

        /// Ports `write` (server-common.h:662 @8642541b2).
        ///
        /// Return: false when the reader is gone (a broken pipe).
        pub fn write(self: *Self, data: T) Error!bool {
            _ = std.c.pthread_mutex_lock(&self.mutex);
            defer _ = std.c.pthread_mutex_unlock(&self.mutex);
            if (self.reader_closed.load(.seq_cst)) {
                return false; // broken pipe
            }
            if (self.max_size > 0) {
                while (self.len() >= self.max_size) {
                    _ = self.pop(); // drop oldest to stay bounded
                }
            }
            try self.queue.append(self.gpa, data);
            _ = std.c.pthread_cond_signal(&self.cv);
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
