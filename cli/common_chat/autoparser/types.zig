//! What the differential analysis reads off a template: where reasoning,
//! content and tool calls start and end, and the shape of a call.
//!
//! # Provenance
//!
//! Ports the data half of `llama.cpp/common/chat-auto-parser.h` (llama.cpp
//! v0.8.0, `8642541b2`): `template_params`, `diff_split`,
//! `compare_variants_result`, the four mode enums with their names, the
//! four analysis sub-structs, the three analyzers' fields, `autoparser`'s
//! fields, `segment_type` and `segment`. The analyzers' and the parser
//! generator's methods are in `analyzer.zig` and `generator.zig`. Each
//! declaration names what it ports and the line it began at.
//!
//! # Ownership
//!
//! Every string here belongs to the arena the analysis ran in.

const std = @import("std");
const json = @import("json");
const caps_mod = @import("../caps.zig");

const Value = json.Value;

/// Ports `template_params` (chat-auto-parser.h:22 @cd959bae9): a probe's
/// render.
pub const TemplateParams = struct {
    messages: Value = .null,
    tools: Value = .null,
    add_generation_prompt: bool = false,
    enable_thinking: bool = true,
    extra_context: ?Value = null,
};

/// Ports `diff_split` (chat-auto-parser.h:30 @cd959bae9).
pub const DiffSplit = struct {
    prefix: []const u8 = "",
    suffix: []const u8 = "",
    left: []const u8 = "",
    right: []const u8 = "",

    /// Ports its `operator`== (chat-auto-parser.h:36 @cd959bae9).
    pub fn eql(a: DiffSplit, b: DiffSplit) bool {
        return std.mem.eql(u8, a.prefix, b.prefix) and std.mem.eql(u8, a.suffix, b.suffix) and
            std.mem.eql(u8, a.left, b.left) and std.mem.eql(u8, a.right, b.right);
    }
};

/// Ports `compare_variants_result` (chat-auto-parser.h:42 @cd959bae9).
pub const CompareVariantsResult = struct {
    diff: DiffSplit,
    output_A: []const u8,
    output_B: []const u8,
};

/// Ports `reasoning_mode` (chat-auto-parser.h:85 @cd959bae9) and its
/// `operator<<` (chat-auto-parser.h:91 @cd959bae9).
pub const ReasoningMode = enum {
    NONE,
    TAG_BASED,
    TOOLS_ONLY,

    pub fn name(self: ReasoningMode) []const u8 {
        return @tagName(self);
    }
};

/// Ports `content_mode` (chat-auto-parser.h:105 @cd959bae9) and its
/// `operator<<` (chat-auto-parser.h:111 @cd959bae9).
pub const ContentMode = enum {
    PLAIN,
    ALWAYS_WRAPPED,
    WRAPPED_WITH_REASONING,

    pub fn name(self: ContentMode) []const u8 {
        return @tagName(self);
    }
};

/// Ports `call_id_position` (chat-auto-parser.h:125 @cd959bae9) and its
/// `operator<<` (chat-auto-parser.h:132 @cd959bae9).
pub const CallIdPosition = enum {
    NONE,
    PRE_FUNC_NAME,
    BETWEEN_FUNC_AND_ARGS,
    POST_ARGS,

    pub fn name(self: CallIdPosition) []const u8 {
        return @tagName(self);
    }
};

/// Ports `tool_format` (chat-auto-parser.h:148 @cd959bae9) and its
/// `operator<<` (chat-auto-parser.h:155 @cd959bae9).
pub const ToolFormat = enum {
    NONE,
    JSON_NATIVE,
    TAG_WITH_JSON,
    TAG_WITH_TAGGED,

    pub fn name(self: ToolFormat) []const u8 {
        return @tagName(self);
    }
};

/// Ports `tool_format_analysis` (chat-auto-parser.h:174 @cd959bae9).
pub const ToolFormatAnalysis = struct {
    mode: ToolFormat = .NONE,
    section_start: []const u8 = "",
    section_end: []const u8 = "",
    per_call_start: []const u8 = "",
    per_call_end: []const u8 = "",
    fun_name_is_key: bool = false,
    tools_array_wrapped: bool = false,
    openai_wrapper_trigger: bool = false,
    function_field: []const u8 = "function",
    name_field: []const u8 = "name",
    args_field: []const u8 = "arguments",
    id_field: []const u8 = "",
    gen_id_field: []const u8 = "",
    parameter_order: std.ArrayList([]const u8) = .empty,
};

/// Ports `tool_function_analysis` (chat-auto-parser.h:194 @cd959bae9).
pub const ToolFunctionAnalysis = struct {
    name_prefix: []const u8 = "",
    name_suffix: []const u8 = "",
    args_separator: []const u8 = "",
    close: []const u8 = "",
};

/// Ports `tool_arguments_analysis` (chat-auto-parser.h:201 @cd959bae9).
pub const ToolArgumentsAnalysis = struct {
    start: []const u8 = "",
    end: []const u8 = "",
    name_prefix: []const u8 = "",
    name_suffix: []const u8 = "",
    value_prefix: []const u8 = "",
    value_suffix: []const u8 = "",
    separator: []const u8 = "",
    tolerate_intertag_whitespace: bool = false,
};

/// Ports `tool_id_analysis` (chat-auto-parser.h:212 @cd959bae9).
pub const ToolIdAnalysis = struct {
    pos: CallIdPosition = .NONE,
    prefix: []const u8 = "",
    suffix: []const u8 = "",
};

/// Ports `analyze_reasoning`'s fields (chat-auto-parser.h:256 @cd959bae9).
pub const AnalyzeReasoning = struct {
    mode: ReasoningMode = .NONE,
    start: []const u8 = "",
    end: []const u8 = "",
};

/// Ports `analyze_content`'s fields (chat-auto-parser.h:283 @cd959bae9).
pub const AnalyzeContent = struct {
    mode: ContentMode = .PLAIN,
    start: []const u8 = "",
    end: []const u8 = "",
    requires_nonnull_content: bool = false,

    /// Ports `is_always_wrapped` (chat-diff-analyzer.cpp:764 @cd959bae9).
    pub fn isAlwaysWrapped(self: *const AnalyzeContent) bool {
        return self.mode == .ALWAYS_WRAPPED and self.start.len != 0 and self.end.len != 0;
    }
};

/// Ports `analyze_tools`'s fields (chat-auto-parser.h:304 @cd959bae9).
pub const AnalyzeTools = struct {
    format: ToolFormatAnalysis = .{},
    function: ToolFunctionAnalysis = .{},
    arguments: ToolArgumentsAnalysis = .{},
    call_id: ToolIdAnalysis = .{},
};

/// Ports `autoparser`'s fields (chat-auto-parser.h:381 @cd959bae9).
pub const Autoparser = struct {
    jinja_caps: caps_mod.Caps = .{},
    user_start: []const u8 = "",
    assistant_start: []const u8 = "",
    reasoning: AnalyzeReasoning = .{},
    content: AnalyzeContent = .{},
    tools: AnalyzeTools = .{},
    analysis_complete: bool = false,
    preserved_tokens: std.ArrayList([]const u8) = .empty,
    additional_stops: std.ArrayList([]const u8) = .empty,
};

/// Ports `segment_type` (chat-auto-parser.h:427 @cd959bae9) and its
/// `operator<<` (chat-auto-parser.h:429 @cd959bae9).
pub const SegmentType = enum {
    TEXT,
    MARKER,

    pub fn name(self: SegmentType) []const u8 {
        return @tagName(self);
    }
};

/// Ports `segment` (chat-auto-parser.h:440 @cd959bae9).
pub const Segment = struct {
    type: SegmentType,
    value: []const u8,

    /// Ports its `operator`== (chat-auto-parser.h:446 @cd959bae9).
    pub fn eql(a: Segment, b: Segment) bool {
        return a.type == b.type and std.mem.eql(u8, a.value, b.value);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
