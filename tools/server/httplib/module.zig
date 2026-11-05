//! Barrel for `tools/server/httplib/`: cpp-httplib's server half, as
//! llama-server's `server-http.cpp` reaches it, and the plain-HTTP client
//! (`client.zig`) the CORS proxy, router mode and the downloader drive.
//!
//! # Provenance
//!
//! **Not a port.** The files it re-exports port parts of
//! `llama.cpp/vendor/cpp-httplib/httplib.cpp` and `httplib.h` (cpp-httplib
//! 0.61.0, vendored at v0.7.0, `620d2225c`); each names its own.

const std = @import("std");

pub const detail = @import("detail.zig");
pub const socket = @import("socket.zig");
pub const message = @import("message.zig");
pub const thread_pool = @import("thread_pool.zig");
pub const server = @import("server.zig");
pub const client = @import("client.zig");

pub const Headers = detail.Headers;
pub const Params = detail.Params;
pub const Request = message.Request;
pub const Response = message.Response;
pub const DataSink = message.DataSink;
pub const ContentProvider = message.ContentProvider;
pub const ThreadPool = thread_pool.ThreadPool;
pub const TaskQueue = thread_pool.TaskQueue;
pub const Server = server.Server;
pub const Handler = server.Handler;
pub const HandlerResponse = server.HandlerResponse;
pub const HandlerWithResponse = server.HandlerWithResponse;
pub const ExceptionHandler = server.ExceptionHandler;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
