//! The string and render helpers the differential analysis is built on:
//! trimming, the marker-aware diff of two renders, segmentation into
//! markers and text, and rendering a probe.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/chat-auto-parser-helpers.h` and
//! `llama.cpp/common/chat-auto-parser-helpers.cpp` (llama.cpp v0.9.0,
//! `cd959bae9`). Each declaration names what it ports and the line it began
//! at.
//!
//! # Ownership
//!
//! Every function takes an arena `a` and returns strings in it or borrowed
//! from its arguments.

const std = @import("std");
const json = @import("json");
const exception = @import("../exception.zig");
const template = @import("../template.zig");
const util = @import("../util.zig");
const log = @import("../log.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Value = json.Value;
const Error = exception.Error;
const Segment = types.Segment;
const DiffSplit = types.DiffSplit;
const isSpace = util.isSpace;

/// Ports `trim_whitespace` (chat-auto-parser-helpers.cpp:12 @cd959bae9).
pub fn trimWhitespace(str: []const u8) []const u8 {
    var start: usize = 0;
    while (start < str.len and isSpace(str[start])) start += 1;
    if (start == str.len) return "";
    var end = str.len - 1;
    while (end > start and isSpace(str[end])) end -= 1;
    return str[start .. end + 1];
}

/// Ports `trim_leading_whitespace` (chat-auto-parser-helpers.cpp:30 @cd959bae9).
pub fn trimLeadingWhitespace(str: []const u8) []const u8 {
    var start: usize = 0;
    while (start < str.len and isSpace(str[start])) start += 1;
    return str[start..];
}

/// Ports `trim_trailing_whitespace` (chat-auto-parser-helpers.cpp:39 @cd959bae9).
pub fn trimTrailingWhitespace(str: []const u8) []const u8 {
    if (str.len == 0) return "";
    var end = str.len - 1;
    while (end > 0 and isSpace(str[end])) end -= 1;
    // If first char is also whitespace, return empty string
    if (end == 0 and isSpace(str[0])) return "";
    return str[0 .. end + 1];
}

/// Ports `trim_trailing_newlines` (chat-auto-parser-helpers.cpp:57 @cd959bae9).
pub fn trimTrailingNewlines(str: []const u8) []const u8 {
    var end = str.len;
    while (end > 0 and str[end - 1] == '\n') end -= 1;
    return str[0..end];
}

/// Ports `common_prefix_len` (chat-auto-parser-helpers.cpp:66 @cd959bae9).
fn commonPrefixLen(left: []const u8, right: []const u8) usize {
    var n: usize = 0;
    const min_len = @min(left.len, right.len);
    while (n < min_len and left[n] == right[n]) n += 1;
    return n;
}

/// Ports `common_suffix_len` (chat-auto-parser-helpers.cpp:75 @cd959bae9).
fn commonSuffixLen(left: []const u8, right: []const u8) usize {
    var n: usize = 0;
    const min_len = @min(left.len, right.len);
    while (n < min_len and left[left.len - 1 - n] == right[right.len - 1 - n]) n += 1;
    return n;
}

fn cat(a: Allocator, x: []const u8, y: []const u8) Error![]const u8 {
    return std.mem.concat(a, u8, &.{ x, y });
}

/// Ports `calculate_diff_split` (chat-auto-parser-helpers.cpp:84 @cd959bae9):
/// the longest common prefix and suffix of two renders, each ending on a
/// marker where one is in reach, and what is left of each between them.
pub fn calculateDiffSplit(a: Allocator, left: []const u8, right: []const u8) Error!DiffSplit {
    var result: DiffSplit = .{};

    const left_seg = try segmentizeMarkers(a, left);
    const right_seg = try segmentizeMarkers(a, right);

    if (left_seg.len == 0) {
        result.right = right;
        return result;
    }
    if (right_seg.len == 0) {
        result.left = left;
        return result;
    }

    var left_start: usize = 0;
    var left_end: usize = left_seg.len - 1;
    var right_start: usize = 0;
    var right_end: usize = right_seg.len - 1;

    var left_fully_consumed = false;
    var right_fully_consumed = false;

    while (left_start != left_end and right_start != right_end) {
        var advanced = false;
        if (left_seg[left_start].eql(right_seg[right_start])) {
            result.prefix = try cat(a, result.prefix, left_seg[left_start].value);
            left_start += 1;
            right_start += 1;
            advanced = true;
        }
        if (left_seg[left_end].eql(right_seg[right_end])) {
            result.suffix = try cat(a, left_seg[left_end].value, result.suffix);
            if (left_start != left_end) {
                left_end -= 1;
            } else {
                left_fully_consumed = true;
            }
            if (right_start != right_end) {
                right_end -= 1;
            } else {
                right_fully_consumed = true;
            }
            advanced = true;
        }
        if (!advanced) break;
    }

    if (left_start == left_end and right_start != right_end) {
        if (left_seg[left_start].eql(right_seg[right_end])) {
            result.suffix = try cat(a, right_seg[right_end].value, result.suffix);
            right_end -= 1;
            left_fully_consumed = true;
        } else if (left_seg[left_start].eql(right_seg[right_start])) {
            result.prefix = try cat(a, result.prefix, right_seg[right_start].value);
            right_start += 1;
            left_fully_consumed = true;
        }
    } else if (right_start == right_end and left_start != left_end) {
        if (left_seg[left_end].eql(right_seg[right_start])) {
            result.suffix = try cat(a, left_seg[left_end].value, result.suffix);
            left_end -= 1;
            right_fully_consumed = true;
        } else if (left_seg[left_start].eql(right_seg[right_start])) {
            result.prefix = try cat(a, result.prefix, left_seg[left_start].value);
            left_start += 1;
            right_fully_consumed = true;
        }
    } else if (left_start == left_end and right_start == right_end and left_seg[left_start].eql(right_seg[right_start]) and left_seg[left_start].type == .MARKER) {
        result.prefix = try cat(a, result.prefix, right_seg[right_start].value);
        left_fully_consumed = true;
        right_fully_consumed = true;
    }

    const can_have_text_suffix = left_seg[left_end].type == .TEXT and right_seg[right_end].type == .TEXT;
    const can_have_text_prefix = right_seg[right_start].type == .TEXT and left_seg[left_start].type == .TEXT;

    const remainder_left = try joinSegments(a, left_seg[left_start..if (left_fully_consumed) left_end else left_end + 1]);
    const remainder_right = try joinSegments(a, right_seg[right_start..if (right_fully_consumed) right_end else right_end + 1]);

    const suffix_len = if (can_have_text_suffix) commonSuffixLen(remainder_left, remainder_right) else 0;
    // avoid overlaps between prefix and suffix
    const prefix_len = if (can_have_text_prefix)
        commonPrefixLen(remainder_left[0 .. remainder_left.len - suffix_len], remainder_right[0 .. remainder_right.len - suffix_len])
    else
        0;

    result.prefix = try cat(a, result.prefix, remainder_left[0..prefix_len]);
    result.suffix = try cat(a, remainder_left[remainder_left.len - suffix_len ..], result.suffix);
    result.left = remainder_left[prefix_len .. remainder_left.len - suffix_len];
    result.right = remainder_right[prefix_len .. remainder_right.len - suffix_len];

    if (result.left.len == 0 and result.right.len == 0) {
        // degenerate case, no diff
        result.prefix = left;
        result.suffix = "";
        // pick prefix = all as representation
    }

    // When left has no unique content (result.left is empty), left is entirely
    // shared with right. The simultaneous prefix/suffix segment matching can
    // incorrectly consume trailing segments of left as suffix when those same
    // segments also appear at the end of right (e.g. "\n" at the end of both
    // the shared content and the generation prompt). This rotates the diff.
    // Fix: if left is a prefix of right, enforce that directly.
    if (result.left.len == 0 and result.right.len != 0 and left.len <= right.len and std.mem.eql(u8, right[0..left.len], left)) {
        result.prefix = left;
        result.suffix = "";
        result.right = right[left.len..];
    }

    return result;
}

/// `std`'s `accumulate` of the segments' values.
fn joinSegments(a: Allocator, segs: []const Segment) Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (segs) |s| try out.appendSlice(a, s.value);
    return out.items;
}

/// Ports `until_common_prefix` (chat-auto-parser-helpers.cpp:207 @cd959bae9):
/// `full` up to the first occurrence of what `left` and `right` begin with.
pub fn untilCommonPrefix(full: []const u8, left: []const u8, right: []const u8) []const u8 {
    const n = commonPrefixLen(left, right);
    if (n == 0) return "";
    const pos = std.mem.indexOf(u8, full, left[0..n]) orelse return "";
    return full[0..pos];
}

/// Ports `after_common_suffix` (chat-auto-parser-helpers.cpp:234 @cd959bae9):
/// `full` after the last occurrence of what `left` and `right` end with.
pub fn afterCommonSuffix(full: []const u8, left: []const u8, right: []const u8) []const u8 {
    const n = commonSuffixLen(left, right);
    if (n == 0) return "";
    const pos = std.mem.lastIndexOf(u8, full, left[left.len - n ..]) orelse return "";
    return full[pos + n ..];
}

/// Ports `segmentize_markers` (chat-auto-parser-helpers.cpp:266 @cd959bae9):
/// `<...>` and `[...]` runs as markers, everything between as text.
///
/// Return: owned by `a`; each value borrowed from `text`.
pub fn segmentizeMarkers(a: Allocator, text: []const u8) Error![]Segment {
    var retval: std.ArrayList(Segment) = .empty;
    var in_marker = false;
    var marker_opener: u8 = 0;
    var last_border: usize = 0;

    for (text, 0..) |ch, cur_pos| {
        if (!in_marker and (ch == '<' or ch == '[')) {
            if (last_border < cur_pos) {
                try retval.append(a, .{ .type = .TEXT, .value = text[last_border..cur_pos] });
            }
            last_border = cur_pos;
            in_marker = true;
            marker_opener = ch;
        } else if (in_marker and ((marker_opener == '<' and ch == '>') or (marker_opener == '[' and ch == ']'))) {
            try retval.append(a, .{ .type = .MARKER, .value = text[last_border .. cur_pos + 1] });
            last_border = cur_pos + 1;
            in_marker = false;
            marker_opener = 0;
        }
    }
    if (last_border < text.len) {
        try retval.append(a, .{ .type = .TEXT, .value = text[last_border..] });
    }
    return retval.items;
}

/// Ports `prune_whitespace_segments` (chat-auto-parser-helpers.cpp:298 @cd959bae9).
pub fn pruneWhitespaceSegments(a: Allocator, segments: []const Segment) Error![]Segment {
    var result: std.ArrayList(Segment) = .empty;
    for (segments) |seg| {
        if (trimWhitespace(seg.value).len != 0) try result.append(a, seg);
    }
    return result.items;
}

/// `ERR_TMPL` (chat-auto-parser-helpers.cpp:310 @cd959bae9): what a failed
/// probe render reads as.
pub const err_tmpl = "#**ERROR**#";

/// Ports `apply_template` (chat-auto-parser-helpers.cpp:312 @cd959bae9): a
/// probe rendered, `enable_thinking` bound in the extra context too, and
/// `err_tmpl` when the template fails.
pub fn applyTemplate(a: Allocator, tmpl: *const template.Template, params: *const types.TemplateParams) Error![]const u8 {
    var tmpl_params: template.GenerationParams = .{};
    tmpl_params.messages = params.messages;
    tmpl_params.tools = params.tools;
    tmpl_params.add_generation_prompt = params.add_generation_prompt;
    tmpl_params.enable_thinking = params.enable_thinking;

    tmpl_params.extra_context = if (params.extra_context) |e| try e.clone(a) else .null;
    const slot = tmpl_params.extra_context.index(a, "enable_thinking") catch |e| return util.fromJson(e);
    slot.* = try Value.from(a, params.enable_thinking);

    const prompt = template.directApply(a, tmpl, &tmpl_params) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        log.dbg("Template application failed: {s}\n", .{exception.what(e)});
        return err_tmpl;
    };
    return prompt.str(a);
}

/// Ports `compare_variants` (chat-auto-parser-helpers.cpp:332 @cd959bae9):
/// two probes rendered and diffed, `params_b` being `params_a` modified as
/// the C++'s lambda modifies its copy.
///
/// Return: null when either render fails.
pub fn compareVariants(a: Allocator, tmpl: *const template.Template, params_a: *const types.TemplateParams, params_b: *const types.TemplateParams) Error!?types.CompareVariantsResult {
    const output_a = try applyTemplate(a, tmpl, params_a);
    const output_b = try applyTemplate(a, tmpl, params_b);
    if (std.mem.eql(u8, output_a, err_tmpl) or std.mem.eql(u8, output_b, err_tmpl)) return null;
    return .{
        .diff = try calculateDiffSplit(a, output_a, output_b),
        .output_A = output_a,
        .output_B = output_b,
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the diff ends on markers, as the header's examples say" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const d = try calculateDiffSplit(a, "<html><body><div></div></body></html>", "<html><body><p>Something</p></body></html>");
    try std.testing.expectEqualStrings("<html><body>", d.prefix);
    try std.testing.expectEqualStrings("</body></html>", d.suffix);
    try std.testing.expectEqualStrings("<div></div>", d.left);
    try std.testing.expectEqualStrings("<p>Something</p>", d.right);
    try std.testing.expectEqualStrings("really want a ", untilCommonPrefix("really want a FUNCTION call", "FUNCTION alpha", "FUNCTION beta"));
    try std.testing.expectEqualStrings(" call", afterCommonSuffix("really want a FUNCTION call", "first FUNCTION", "second FUNCTION"));
}
