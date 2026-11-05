//! Barrel for `cli/common_chat/`: upstream's `common/chat.h` -- the chat
//! data model and the chat side of the PEG parser now, the template
//! application and output parsing of Stage 9 step 4 later.
//!
//! # Provenance
//!
//! **Not a port.** The files it re-exports port parts of
//! `llama.cpp/common/chat.h` and `llama.cpp/common/chat.cpp`, and all of
//! `llama.cpp/common/chat-peg-parser.h` and `chat-peg-parser.cpp`; each
//! names its own.
//!
//! It imports `json` by module name, and `chat_peg.zig` also `peg` and
//! `llama`: whatever builds this module and reaches `chat_peg` supplies
//! them.

const std = @import("std");

pub const exception = @import("exception.zig");
pub const msg = @import("msg.zig");
pub const chat_peg = @import("chat_peg.zig");
pub const log = @import("log.zig");
pub const caps = @import("caps.zig");
pub const template = @import("template.zig");
pub const util = @import("util.zig");
pub const params = @import("params.zig");
pub const parsers = @import("parsers/module.zig");
pub const autoparser = @import("autoparser/module.zig");
pub const templates = @import("templates.zig");
pub const parse = @import("parse.zig");
pub const prompt = @import("prompt.zig");
pub const input = @import("input.zig");
pub const session = @import("session.zig");

pub const Error = exception.Error;
pub const what = exception.what;

pub const Msg = msg.Msg;
pub const ToolCall = msg.ToolCall;
pub const ContentPart = msg.ContentPart;
pub const MsgDiff = msg.MsgDiff;
pub const Role = msg.Role;
pub const MsgSpan = msg.MsgSpan;
pub const MsgSpans = msg.MsgSpans;
pub const MsgDelimiter = msg.MsgDelimiter;
pub const MsgDelimiters = msg.MsgDelimiters;
pub const Tool = msg.Tool;
pub const ToolChoice = msg.ToolChoice;
pub const Format = msg.Format;
pub const Continuation = msg.Continuation;
pub const ReasoningFormat = msg.ReasoningFormat;
pub const ContentCaps = msg.ContentCaps;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    _ = exception;
    _ = msg;
    _ = chat_peg;
    _ = log;
    _ = caps;
    _ = template;
    _ = util;
    _ = params;
    _ = parsers;
    _ = autoparser;
    _ = templates;
    _ = parse;
    _ = prompt;
    _ = input;
    _ = session;
}
