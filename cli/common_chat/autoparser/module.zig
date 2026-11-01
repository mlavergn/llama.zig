//! Barrel for `cli/common_chat/autoparser/`: reading a template's format off
//! the differences between renders of it, and building a parser from what
//! was read.
//!
//! # Provenance
//!
//! **Not a port.** The files it re-exports port
//! `llama.cpp/common/chat-auto-parser.h`,
//! `llama.cpp/common/chat-auto-parser-helpers.h` and `.cpp`,
//! `llama.cpp/common/chat-diff-analyzer.cpp` and
//! `llama.cpp/common/chat-auto-parser-generator.cpp`; each names its own.

const std = @import("std");

pub const types = @import("types.zig");
pub const helpers = @import("helpers.zig");
pub const analyzer = @import("analyzer.zig");
pub const generator = @import("generator.zig");

pub const Autoparser = types.Autoparser;
pub const ReasoningMode = types.ReasoningMode;
pub const ContentMode = types.ContentMode;
pub const ToolFormat = types.ToolFormat;
pub const CallIdPosition = types.CallIdPosition;
pub const DiffSplit = types.DiffSplit;
pub const Segment = types.Segment;
pub const SegmentType = types.SegmentType;
pub const TemplateParams = types.TemplateParams;

pub const analyzeTemplate = analyzer.analyzeTemplate;
pub const generateParser = generator.generateParser;
pub const generateParserAnalyzed = generator.generateParserAnalyzed;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = types;
    _ = helpers;
    _ = analyzer;
    _ = generator;
}
