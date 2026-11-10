//! cpp-httplib's `Request` and `Response` as the server fills them, and the
//! `detail` code that reads one off a stream and writes the other back:
//! header lines, request bodies (`Content-Length`, chunked, unframed),
//! `multipart/form-data`, byte ranges, the chunked response writer.
//!
//! # Provenance
//!
//! Ports parts of `llama.cpp/vendor/cpp-httplib/httplib.cpp` and
//! `llama.cpp/vendor/cpp-httplib/httplib.h` (cpp-httplib 0.64.0, vendored
//! at v0.9.0, `cd959bae9`). Each declaration names what it ports and the
//! line it began at.
//!
//! # Memory
//!
//! A `Request` and its `Response` live in one arena for one request, as the
//! C's two stack objects live in `process_request`'s frame: every string in
//! them is the arena's, and nothing in them is freed before it is.

const std = @import("std");
const libcxx = @import("llama").libcxx;
const detail = @import("detail.zig");
const socket = @import("socket.zig");

const Headers = detail.Headers;
const Params = detail.Params;
const Range = detail.Range;
const SocketStream = socket.SocketStream;
const LineReader = socket.LineReader;

/// Ports `FormData` (httplib.h:1514 @cd959bae9): one file part.
pub const FormData = struct {
    name: []const u8 = "",
    content: std.ArrayList(u8) = .empty,
    filename: []const u8 = "",
    content_type: []const u8 = "",
    headers: Headers = .{},
};

/// Ports `FormField` (httplib.h:1522 @cd959bae9): one text part.
pub const FormField = struct {
    name: []const u8 = "",
    content: std.ArrayList(u8) = .empty,
    headers: Headers = .{},
};

/// Ports `MultipartFormData` (httplib.h:1538 @cd959bae9): the parts of a
/// `multipart/form-data` body, in the order sent.
pub const MultipartFormData = struct {
    fields: detail.OrderedMultimap(FormField, false) = .{},
    files: detail.OrderedMultimap(FormData, false) = .{},
};

/// Ports `Request` (httplib.h:1811 @cd959bae9): the members the server
/// fills and reads.
pub const Request = struct {
    a: std.mem.Allocator,
    method: []const u8 = "",
    path: []const u8 = "",
    matched_route: []const u8 = "",
    params: Params = .{},
    headers: Headers = .{},
    trailers: Headers = .{},
    body: std.ArrayList(u8) = .empty,
    remote_addr: []const u8 = "",
    remote_port: c_int = -1,
    local_addr: []const u8 = "",
    local_port: c_int = -1,
    version: []const u8 = "",
    target: []const u8 = "",
    form: MultipartFormData = .{},
    ranges: std.ArrayList(Range) = .empty,
    /// `path_params`: an `unordered_map` in the C, whose readers copy it
    /// into a `std::map`, so its order is never seen.
    path_params: std.ArrayList(Params.Entry) = .empty,
    /// The socket `is_connection_closed` polls; -1 until `process_request`
    /// sets it up, where the C's default answers "closed".
    sock: socket.Socket = -1,
    /// `body_read_started_`: a second read would start where the first stopped.
    body_read_started: bool = false,
    body_consumed: bool = false,
    /// `100 Continue` is owed and not yet sent: `read_content_core` sends
    /// it when the body is actually read.
    expect_100_continue_pending: bool = false,

    /// Ports `has_header` (httplib.cpp:7183 @cd959bae9), a member of
    /// `Request`.
    pub fn hasHeader(self: *const Request, key: []const u8) bool {
        return detail.hasHeader(&self.headers, key);
    }

    /// Ports `get_header_value` (httplib.cpp:7187 @cd959bae9), a member of
    /// `Request`.
    pub fn getHeaderValue(self: *const Request, key: []const u8) []const u8 {
        return detail.getHeaderValue(&self.headers, key);
    }

    /// `get_header_value_u64` with its defaults, a member of `Request`.
    pub fn getHeaderValueU64(self: *const Request, key: []const u8) usize {
        var invalid = false;
        return detail.getHeaderValueU64(&self.headers, key, 0, &invalid);
    }

    /// Ports `is_multipart_form_data` (httplib.cpp:7238 @cd959bae9), a
    /// member of `Request`.
    pub fn isMultipartFormData(self: *const Request) error{OutOfMemory}!bool {
        const content_type = self.getHeaderValue("Content-Type");
        return detail.caseIgnoreEqual(try detail.extractMediaType(self.a, content_type, null), "multipart/form-data");
    }

    /// `is_connection_closed`, as `process_request` sets it up: whether
    /// the peer has gone.
    pub fn isConnectionClosed(self: *const Request) bool {
        if (self.sock < 0) return true;
        return !socket.isSocketAlive(self.sock);
    }
};

/// `DataSink` (httplib.h:1563 @cd959bae9): where a content provider writes,
/// bound by the writer that runs it.
pub const DataSink = struct {
    ctx: *anyopaque,
    write_fn: *const fn (ctx: *anyopaque, data: []const u8) bool,
    done_fn: *const fn (ctx: *anyopaque) void,

    pub fn write(self: *DataSink, data: []const u8) bool {
        return self.write_fn(self.ctx, data);
    }

    pub fn done(self: *DataSink) void {
        self.done_fn(self.ctx);
    }
};

/// `ContentProviderWithoutLength`, behind `ContentProviderAdapter`
/// (httplib.cpp:5664 @cd959bae9): called with the offset written so far
/// and a sink. An error is an exception thrown out of the provider.
pub const ContentProvider = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, offset: usize, sink: *DataSink) anyerror!bool,
};

/// `ContentProviderResourceReleaser`: told, as the response goes away,
/// whether the provider ran to the end.
pub const Releaser = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque, success: bool) void,
};

/// Ports `Response` (httplib.h:1887 @cd959bae9): the members the server
/// fills. `content_length_` stays 0 and the content coding `None`: the
/// server's handlers set neither a known-length provider nor a file.
pub const Response = struct {
    a: std.mem.Allocator,
    status: c_int = -1,
    headers: Headers = .{},
    body: std.ArrayList(u8) = .empty,
    content_length: usize = 0,
    content_provider: ?ContentProvider = null,
    releaser: ?Releaser = null,
    is_chunked_content_provider: bool = false,
    content_provider_success: bool = false,

    /// `~Response`: tells the releaser how the provider ended.
    pub fn deinit(self: *Response) void {
        if (self.releaser) |r| {
            self.releaser = null;
            r.func(r.ctx, self.content_provider_success);
        }
    }

    /// Ports `clear_content_provider` (httplib.cpp:7341 @cd959bae9): a
    /// response has one content source, so each content setter calls this
    /// and the one called last wins. No file content is ever set here.
    pub fn clearContentProvider(self: *Response) void {
        const releaser = self.releaser;
        const success = self.content_provider_success;
        self.content_length = 0;
        self.content_provider = null;
        self.releaser = null;
        self.is_chunked_content_provider = false;
        self.content_provider_success = false;
        if (releaser) |r| r.func(r.ctx, success);
    }

    /// Ports `clear_content` (httplib.cpp:7358 @cd959bae9).
    pub fn clearContent(self: *Response) void {
        self.body.clearRetainingCapacity();
        self.clearContentProvider();
    }

    /// Ports `set_content_type` (httplib.cpp:7364 @cd959bae9): the
    /// Content-Type of the old content goes with it.
    fn setContentType(self: *Response, content_type: []const u8) error{OutOfMemory}!void {
        _ = self.headers.eraseKey("Content-Type");
        try self.setHeader("Content-Type", content_type);
    }

    /// Ports `prefer_assigned_body` (httplib.cpp:7371 @cd959bae9): `body`
    /// is public, so it can be assigned directly after a provider was set.
    /// It came last, so it wins.
    pub fn preferAssignedBody(self: *Response) void {
        if (self.body.items.len != 0 and self.content_provider != null) self.clearContentProvider();
    }

    /// Ports `has_header` (httplib.cpp:7444 @cd959bae9), a member of
    /// `Response`: unlike `Request`'s, no name is prohibited.
    pub fn hasHeader(self: *const Response, key: []const u8) bool {
        return self.headers.find(key) != null;
    }

    /// Ports `get_header_value` (httplib.cpp:7448 @cd959bae9), a member of
    /// `Response`.
    pub fn getHeaderValue(self: *const Response, key: []const u8) []const u8 {
        return detail.getHeaderValue(&self.headers, key);
    }

    /// Ports `set_header` (httplib.cpp:7458 @cd959bae9), a member of
    /// `Response`: appends, so a name set twice is sent twice; an invalid
    /// name or value is dropped. Both strings are copied.
    pub fn setHeader(self: *Response, key: []const u8, val: []const u8) error{OutOfMemory}!void {
        if (!detail.isFieldValid(key, val)) return;
        _ = try self.headers.emplace(self.a, try self.a.dupe(u8, key), try self.a.dupe(u8, val));
    }

    /// Ports `set_content` (httplib.cpp:7486 @cd959bae9), a member of
    /// `Response`, and the overload over a string (httplib.cpp:7493
    /// @cd959bae9): replaces the body and every `Content-Type`.
    pub fn setContent(self: *Response, s: []const u8, content_type: []const u8) error{OutOfMemory}!void {
        self.body.clearRetainingCapacity();
        try self.body.appendSlice(self.a, s); // Before the releaser runs; it may own `s`.
        self.clearContentProvider();
        try self.setContentType(content_type);
    }

    /// Ports `set_chunked_content_provider` (httplib.cpp:7526 @cd959bae9),
    /// a member of `Response`.
    pub fn setChunkedContentProvider(self: *Response, content_type: []const u8, provider: ContentProvider, releaser: ?Releaser) error{OutOfMemory}!void {
        self.clearContent();
        try self.setContentType(content_type);
        self.content_provider = provider;
        self.releaser = releaser;
        self.is_chunked_content_provider = true;
    }
};

// -----------------------------------------------------------------------------
// Header lines

/// Ports `parse_header` (httplib.cpp:1250 @cd959bae9): `name: value`, the
/// value's surrounding spaces and tabs dropped.
///
/// Return: the two, borrowed from `line`, or null where the C returns false.
pub fn parseHeader(line: []const u8) ?struct { key: []const u8, val: []const u8 } {
    var end = line.len;
    while (0 < end and detail.isSpaceOrTab(line[end - 1])) end -= 1;
    var p: usize = 0;
    while (p < end and line[p] != ':') p += 1;
    const name = line[0..p];
    if (!detail.isFieldName(name)) return null;
    if (p == end) return null;
    const key_end = p;
    p += 1;
    while (p < end and detail.isSpaceOrTab(line[p])) p += 1;
    if (key_end == 0) return null;
    const val = line[p..end];
    if (!detail.isFieldValue(val)) return null;
    return .{ .key = line[0..key_end], .val = val };
}

/// Ports `read_headers` (httplib.cpp:3591 @cd959bae9): field lines up to
/// the blank one, each at most `CPPHTTPLIB_HEADER_MAX_LENGTH`, at most
/// `CPPHTTPLIB_HEADER_MAX_COUNT` of them, and no two `Content-Length`s that
/// differ.
pub fn readHeaders(a: std.mem.Allocator, strm: *SocketStream, headers: *Headers) error{OutOfMemory}!bool {
    var buf: [2048]u8 = undefined;
    var lr = LineReader.init(a, strm, &buf);
    defer lr.deinit();
    var header_count: usize = 0;
    while (true) {
        if (!try lr.getline()) return false;
        const line = lr.line();
        if (lr.endWithCrlf()) {
            if (line.len == 2) break;
        } else {
            continue; // Skip invalid line.
        }
        if (line.len > detail.header_max_length) return false;
        if (header_count >= detail.header_max_count) return false;
        const h = parseHeader(line[0 .. line.len - 2]) orelse return false;
        _ = try headers.emplace(a, try a.dupe(u8, h.key), try a.dupe(u8, h.val));
        header_count += 1;
    }
    var it = headers.find("Content-Length") orelse return true;
    const first_val = headers.entries.items[it].value;
    while (headers.nextOf("Content-Length", it)) |j| : (it = j) {
        if (!std.mem.eql(u8, headers.entries.items[j].value, first_val)) return false;
    }
    return true;
}

/// The names `parse_trailers` never takes from a trailer, whatever
/// `Trailer` declares.
const prohibited_trailers = [_][]const u8{
    "transfer-encoding", "content-length",     "host",                "authorization",
    "www-authenticate",  "proxy-authenticate", "proxy-authorization", "cookie",
    "set-cookie",        "cache-control",      "expect",              "max-forwards",
    "pragma",            "range",              "te",                  "age",
    "expires",           "date",               "location",            "retry-after",
    "vary",              "warning",            "content-encoding",    "content-type",
    "content-range",     "trailer",
};

/// Ports `parse_trailers` (httplib.cpp:1295 @cd959bae9): field lines after
/// the last chunk, until the blank one, kept in `dest` only where
/// `Trailer` declared them. At most `CPPHTTPLIB_HEADER_MAX_COUNT` names are
/// declared, and every field received counts toward the same limit.
fn parseTrailers(a: std.mem.Allocator, lr: *LineReader, dest: *Headers, src_headers: *const Headers) error{OutOfMemory}!bool {
    var declared: std.ArrayList([]const u8) = .empty;
    defer declared.deinit(a);
    const trailer_header = try detail.getCombinedHeaderValue(a, src_headers, "Trailer");
    if (trailer_header.len != 0) {
        const Ctx = struct {
            a: std.mem.Allocator,
            declared: *std.ArrayList([]const u8),
            pub fn call(self: *@This(), key: []const u8) error{OutOfMemory}!void {
                // A legitimate message declares only a handful of trailers. Cap the
                // set so a peer cannot grow it without bound.
                if (self.declared.items.len >= detail.header_max_count) return;
                for (prohibited_trailers) |p| if (detail.caseIgnoreEqual(p, key)) return;
                for (self.declared.items) |d| if (detail.caseIgnoreEqual(d, key)) return;
                try self.declared.append(self.a, key);
            }
        };
        var ctx: Ctx = .{ .a = a, .declared = &declared };
        try detail.split(trailer_header, ',', std.math.maxInt(usize), &ctx);
    }
    var trailer_header_count: usize = 0;
    while (!std.mem.eql(u8, lr.cstr(), "\r\n")) {
        const line = lr.line();
        if (line.len > detail.header_max_length) return false;
        if (trailer_header_count >= detail.header_max_count) return false;
        if (line.len < 2) return false;
        const h = parseHeader(line[0 .. line.len - 2]) orelse return false;
        for (declared.items) |d| {
            if (detail.caseIgnoreEqual(d, h.key)) {
                _ = try dest.emplace(a, try a.dupe(u8, h.key), try a.dupe(u8, h.val));
                break;
            }
        }
        // Count every received trailer field, not only the declared ones stored in
        // dest, so undeclared fields cannot keep this loop running past the limit.
        trailer_header_count += 1;
        if (!try lr.getline()) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Request bodies

/// Ports `ReadContentResult` (httplib.cpp:3767 @cd959bae9).
const ReadContentResult = enum { success, payload_too_large, err };

/// Ports `ChunkedDecoder` (httplib.h:4061 @cd959bae9): a chunked body's
/// payload, one chunk-size line and its data at a time.
const ChunkedDecoder = struct {
    strm: *SocketStream,
    chunk_remaining: usize = 0,
    finished: bool = false,
    line_buf: [64]u8 = undefined,
    overhead: usize = 0,

    /// Ports `read_payload` (httplib.cpp:11268 @cd959bae9), a member of
    /// `ChunkedDecoder`. The chunk-size line is read up to its terminator,
    /// never past it: only a chunk extension of visible characters, spaces
    /// and tabs may follow the size. The bytes around the data are counted
    /// apart, and more of them than one header field fails the body.
    ///
    /// Return: bytes read into `buf`, 0 after the last chunk, -1 on a
    /// framing error.
    fn readPayload(self: *ChunkedDecoder, a: std.mem.Allocator, buf: []u8) error{OutOfMemory}!isize {
        if (self.finished) return 0;
        if (self.chunk_remaining == 0) {
            var lr = LineReader.init(a, self.strm, &self.line_buf);
            defer lr.deinit();
            if (!try lr.getline()) return -1;

            // Everything below is bounded by eol rather than by the buffer's NUL, so
            // the line terminator is never mistaken for line content.
            const line = lr.line();
            var eol = line.len;
            if (lr.endWithCrlf()) {
                eol -= 2;
            } else if (eol != 0 and line[eol - 1] == '\n') {
                // Only reachable under CPPHTTPLIB_ALLOW_LF_AS_LINE_TERMINATOR
                eol -= 1;
            }

            // RFC 9112 §7.1: chunk-size = 1*HEXDIG
            const s = line;
            var p: usize = 0;
            if (p == eol or detail.isHex(s[p]) == null) return -1;
            var chunk_len: usize = 0;
            while (p < eol) : (p += 1) {
                const v = detail.isHex(s[p]) orelse break;
                if (chunk_len > (std.math.maxInt(usize) >> 4)) return -1;
                chunk_len = (chunk_len << 4) | v;
            }
            while (p < eol and detail.isSpaceOrTab(s[p])) p += 1;

            // RFC 9112 §7.1.1: only a chunk-ext may sit between the size and the line
            // terminator, built from tokens and quoted-strings, so never a CR, LF or
            // any other control character (request smuggling).
            if (p < eol and s[p] != ';') return -1;
            while (p < eol) : (p += 1) {
                if (!detail.isSpaceOrTab(s[p]) and !detail.isFieldVchar(s[p])) return -1;
            }

            // The bytes around the data are no payload, so they are counted on their
            // own, as Go does: each chunk may spend 16 bytes and twice its data on
            // them, which lets a stream send a byte at a time, and what goes past that
            // in padding, leading zeros and extensions builds up to one header field.
            self.overhead += line.len + 2;
            self.overhead = if (self.overhead > 16) self.overhead - 16 else 0;
            self.overhead = if (chunk_len > self.overhead / 2) 0 else self.overhead - 2 * chunk_len;
            if (self.overhead > detail.header_max_length) return -1;

            if (chunk_len == 0) {
                self.chunk_remaining = 0;
                self.finished = true;
                return 0;
            }
            self.chunk_remaining = chunk_len;
        }
        const to_read = @min(self.chunk_remaining, buf.len);
        const n = self.strm.read(buf[0..to_read]);
        if (n <= 0) return -1;
        self.chunk_remaining -= @intCast(n);
        if (self.chunk_remaining == 0) {
            var lr = LineReader.init(a, self.strm, &self.line_buf);
            defer lr.deinit();
            if (!try lr.getline()) return -1;
            if (!std.mem.eql(u8, lr.cstr(), "\r\n")) return -1;
        }
        return n;
    }

    /// Ports `parse_trailers_into` (httplib.cpp:11359 @cd959bae9), a member
    /// of `ChunkedDecoder`.
    fn parseTrailersInto(self: *ChunkedDecoder, a: std.mem.Allocator, dest: *Headers, src_headers: *const Headers) error{OutOfMemory}!bool {
        var lr = LineReader.init(a, self.strm, &self.line_buf);
        defer lr.deinit();
        if (!try lr.getline()) return false;
        return parseTrailers(a, &lr, dest, src_headers);
    }
};

/// Ports `read_content_with_length` (httplib.cpp:3773 @cd959bae9), with
/// the `BodyReader` it reads through (`BodyReader::read`, httplib.cpp:7579
/// @cd959bae9): `len` bytes, failing past `payload_max_length`.
fn readContentWithLength(strm: *SocketStream, len: usize, out: anytype, payload_max_length: usize) !ReadContentResult {
    var buf: [detail.recv_bufsiz]u8 = undefined;
    var bytes_read: usize = 0;
    var r: usize = 0;
    while (r < len) {
        const to_read = @min(len - r, detail.recv_bufsiz);
        // BodyReader::read
        if (bytes_read >= len) return .err;
        const n = strm.read(buf[0..@min(to_read, len - bytes_read)]);
        if (n <= 0) return .err;
        bytes_read += @intCast(n);
        if (payload_max_length > 0 and bytes_read > payload_max_length) return .payload_too_large;
        if (!try out.receive(buf[0..@intCast(n)])) return .err;
        r += @intCast(n);
    }
    return .success;
}

/// Ports `read_content_without_length` (httplib.cpp:3815 @cd959bae9): to
/// end of stream. A `payload_max_length` of 0 is no limit, since 0.60.0.
fn readContentWithoutLength(strm: *SocketStream, payload_max_length: usize, out: anytype) !ReadContentResult {
    var buf: [detail.recv_bufsiz]u8 = undefined;
    var r: usize = 0;
    while (true) {
        const n = strm.read(&buf);
        if (n == 0) return .success;
        if (n < 0) return .err;
        const un: usize = @intCast(n);
        if (payload_max_length > 0 and
            (r > payload_max_length or payload_max_length - r < un)) return .payload_too_large;
        if (!try out.receive(buf[0..un])) return .err;
        r += un;
    }
}

/// Ports `read_content_chunked` (httplib.cpp:3841 @cd959bae9): the chunks,
/// then the trailers into `req.trailers`. A `payload_max_length` of 0 is
/// no limit, since 0.60.0.
fn readContentChunked(strm: *SocketStream, req: anytype, payload_max_length: usize, out: anytype) !ReadContentResult {
    var dec: ChunkedDecoder = .{ .strm = strm };
    var buf: [detail.recv_bufsiz]u8 = undefined;
    var total_len: usize = 0;
    while (true) {
        const n = try dec.readPayload(req.a, &buf);
        if (n < 0) return .err;
        if (n == 0) {
            if (!try dec.parseTrailersInto(req.a, &req.trailers, &req.headers)) return .err;
            return .success;
        }
        const un: usize = @intCast(n);
        if (payload_max_length > 0 and
            (total_len > payload_max_length or payload_max_length - total_len < un)) return .payload_too_large;
        if (!try out.receive(buf[0..un])) return .err;
        total_len += un;
    }
}

/// Ports `read_content` (httplib.cpp:3982 @cd959bae9) over a request, with
/// `prepare_content_receiver` (httplib.cpp:3927 @cd959bae9): a body framed
/// by chunks, by `Content-Length`, or by the end of the stream. With
/// `decompress`, a coding this build cannot decode answers 415 -- no
/// decompressor is compiled in (`create_decompressor`, httplib.cpp:3455
/// @cd959bae9), so every one it knows is such a coding.
///
/// `out` has `receive([]const u8) !bool`. `req` is any message with `a`,
/// `headers` and `trailers`: a server's request, or a client's response
/// (`client.zig`), as the C's template takes either.
pub fn readContent(strm: *SocketStream, req: anytype, payload_max_length: usize, status: *c_int, out: anytype, decompress: bool) !bool {
    if (decompress) {
        const encoding = try detail.getCombinedHeaderValue(req.a, &req.headers, "Content-Encoding");
        if (encoding.len != 0 and detail.isKnownContentEncoding(encoding)) {
            status.* = detail.StatusCode.unsupported_media_type_415;
            return false;
        }
    }
    var exceed_payload_max_length = false;
    var ret = true;
    if (detail.isChunkedTransferEncoding(&req.headers)) {
        switch (try readContentChunked(strm, req, payload_max_length, out)) {
            .success => ret = true,
            .payload_too_large => {
                exceed_payload_max_length = true;
                ret = false;
            },
            .err => ret = false,
        }
    } else if (!detail.hasHeader(&req.headers, "Content-Length")) {
        switch (try readContentWithoutLength(strm, payload_max_length, out)) {
            .success => ret = true,
            .payload_too_large => {
                exceed_payload_max_length = true;
                ret = false;
            },
            .err => ret = false,
        }
    } else {
        var is_invalid_value = false;
        const len = detail.getHeaderValueU64(&req.headers, "Content-Length", std.math.maxInt(usize), &is_invalid_value);
        if (is_invalid_value) {
            ret = false;
        } else if (len > 0) {
            const result = try readContentWithLength(strm, len, out, payload_max_length);
            ret = result == .success;
            if (result == .payload_too_large) exceed_payload_max_length = true;
        }
    }
    if (!ret) status.* = if (exceed_payload_max_length) detail.StatusCode.payload_too_large_413 else detail.StatusCode.bad_request_400;
    return ret;
}

/// Ports `has_framed_body` (httplib.cpp:5538 @cd959bae9).
pub fn hasFramedBody(req: *const Request) bool {
    return detail.isChunkedTransferEncoding(&req.headers) or req.getHeaderValueU64("Content-Length") > 0;
}

/// Ports `expect_content` (httplib.cpp:5543 @cd959bae9).
pub fn expectContent(req: *const Request) bool {
    const m = req.method;
    if (std.mem.eql(u8, m, "POST") or std.mem.eql(u8, m, "PUT") or std.mem.eql(u8, m, "PATCH") or std.mem.eql(u8, m, "DELETE")) return true;
    return hasFramedBody(req);
}

/// Ports `FormDataParser` (httplib.cpp:4763 @cd959bae9): a
/// `multipart/form-data` body, fed in pieces, as a state machine over its
/// own buffer.
pub const FormDataParser = struct {
    a: std.mem.Allocator,
    dash_boundary_crlf: []const u8 = "",
    crlf_dash_boundary: []const u8 = "",
    state: usize = 0,
    is_valid: bool = false,
    file: FormData = .{},
    header_count: usize = 0,
    header_searched: usize = 0,
    buf: std.ArrayList(u8) = .empty,
    buf_spos: usize = 0,
    buf_epos: usize = 0,

    const dash = "--";
    const crlf = "\r\n";

    /// `set_boundary`, a member of `FormDataParser`.
    pub fn setBoundary(self: *FormDataParser, boundary: []const u8) error{OutOfMemory}!void {
        self.dash_boundary_crlf = try std.mem.concat(self.a, u8, &.{ dash, boundary, crlf });
        self.crlf_dash_boundary = try std.mem.concat(self.a, u8, &.{ crlf, dash, boundary });
    }

    fn bufSize(self: *const FormDataParser) usize {
        return self.buf_epos - self.buf_spos;
    }

    fn bufData(self: *const FormDataParser) []const u8 {
        return self.buf.items[self.buf_spos..self.buf_epos];
    }

    fn bufErase(self: *FormDataParser, size: usize) void {
        self.buf_spos += size;
    }

    fn bufStartWith(self: *const FormDataParser, s: []const u8) bool {
        return std.mem.startsWith(u8, self.bufData(), s);
    }

    /// `buf_find`, a member of `FormDataParser`.
    ///
    /// Return: the offset of `s` in what is buffered, or the buffered size.
    fn bufFind(self: *const FormDataParser, s: []const u8) usize {
        return self.bufFindFrom(s, 0);
    }

    fn bufFindFrom(self: *const FormDataParser, s: []const u8, from: usize) usize {
        if (from >= self.bufSize()) return self.bufSize();
        return std.mem.findPos(u8, self.bufData(), from, s) orelse self.bufSize();
    }

    /// `buf_append`, a member of `FormDataParser`.
    fn bufAppend(self: *FormDataParser, data: []const u8) error{OutOfMemory}!void {
        const remaining_size = self.bufSize();
        if (remaining_size > 0 and self.buf_spos > 0) {
            std.mem.copyForwards(u8, self.buf.items[0..remaining_size], self.buf.items[self.buf_spos..][0..remaining_size]);
        }
        self.buf_spos = 0;
        self.buf_epos = remaining_size;
        if (remaining_size + data.len > self.buf.items.len) try self.buf.resize(self.a, remaining_size + data.len);
        @memcpy(self.buf.items[self.buf_epos..][0..data.len], data);
        self.buf_epos += data.len;
    }

    fn startWithCaseIgnore(a: []const u8, b: []const u8, offset: usize) bool {
        if (a.len < offset + b.len) return false;
        for (b, 0..) |ch, i| if (detail.toLower(a[offset + i]) != detail.toLower(ch)) return false;
        return true;
    }

    /// `parse_content_disposition`, a member of `FormDataParser`.
    ///
    /// Return: the parameters after `form-data;`, or null.
    fn parseContentDisposition(header: []const u8) ?[]const u8 {
        const prefix = "Content-Disposition:";
        if (!startWithCaseIgnore(header, prefix, 0)) return null;
        var pos: usize = prefix.len;
        while (pos < header.len and (header[pos] == ' ' or header[pos] == '\t')) pos += 1;
        const form_data = "form-data;";
        if (!startWithCaseIgnore(header, form_data, pos)) return null;
        pos += form_data.len;
        while (pos < header.len and (header[pos] == ' ' or header[pos] == '\t')) pos += 1;
        return header[pos..];
    }

    /// `parse`, a member of `FormDataParser`. `cb` has
    /// `header(*const FormData) !bool` and `content([]const u8) !bool`.
    ///
    /// Return: false where the body cannot be valid.
    pub fn parse(self: *FormDataParser, data: []const u8, cb: anytype) !bool {
        if (self.state == 5) return true;
        try self.bufAppend(data);
        while (self.bufSize() > 0) {
            switch (self.state) {
                0 => {
                    const pos = self.bufFind(self.dash_boundary_crlf);
                    if (pos == self.bufSize()) {
                        const keep = self.dash_boundary_crlf.len - 1;
                        if (self.bufSize() > keep) self.bufErase(self.bufSize() - keep);
                        return true;
                    }
                    self.bufErase(pos + self.dash_boundary_crlf.len);
                    self.state = 1;
                },
                1 => {
                    self.file = .{};
                    self.header_count = 0;
                    self.header_searched = 0;
                    self.state = 2;
                },
                2 => {
                    // Resume where the last call stopped looking, so that a line arriving
                    // a few bytes at a time is not searched from its start each time.
                    var pos = self.bufFindFrom(crlf, self.header_searched);
                    if (pos > detail.header_max_length) return false;
                    while (pos < self.bufSize()) {
                        if (pos == 0) {
                            if (!try cb.header(&self.file)) {
                                self.is_valid = false;
                                return false;
                            }
                            self.bufErase(crlf.len);
                            self.state = 3;
                            break;
                        }
                        if (self.header_count >= detail.header_max_count) {
                            self.is_valid = false;
                            return false;
                        }
                        self.header_count += 1;
                        const header = try self.a.dupe(u8, self.bufData()[0..pos]);
                        const h = parseHeader(header) orelse {
                            self.is_valid = false;
                            return false;
                        };
                        _ = try self.file.headers.emplace(self.a, h.key, h.val);

                        const header_content_type = "Content-Type:";
                        if (startWithCaseIgnore(header, header_content_type, 0)) {
                            self.file.content_type = detail.trimCopy(header[header_content_type.len..]);
                        } else if (parseContentDisposition(header)) |disposition_params| {
                            var params: Params = .{};
                            try detail.parseDispositionParams(self.a, disposition_params, &params);
                            self.file.name = params.get("name", 0) orelse {
                                self.is_valid = false;
                                return false;
                            };
                            if (params.get("filename", 0)) |f| self.file.filename = f;
                            if (params.get("filename*", 0)) |val| {
                                const utf8_prefix = "UTF-8''";
                                if (val.len > utf8_prefix.len and startWithCaseIgnore(val, utf8_prefix, 0)) {
                                    self.file.filename = try detail.decodePathComponent(self.a, val[utf8_prefix.len..]);
                                } else {
                                    self.is_valid = false;
                                    return false;
                                }
                            }
                        }
                        self.bufErase(pos + crlf.len);
                        pos = self.bufFind(crlf);
                    }
                    if (self.state != 3) {
                        // The CR of the CRLF may be the last byte so far.
                        self.header_searched = if (self.bufSize() > 0) self.bufSize() - 1 else 0;
                        return true;
                    }
                },
                3 => {
                    if (self.crlf_dash_boundary.len > self.bufSize()) return true;
                    const pos = self.bufFind(self.crlf_dash_boundary);
                    if (pos < self.bufSize()) {
                        if (!try cb.content(self.bufData()[0..pos])) {
                            self.is_valid = false;
                            return false;
                        }
                        self.bufErase(pos + self.crlf_dash_boundary.len);
                        self.state = 4;
                    } else {
                        const len = self.bufSize() - self.crlf_dash_boundary.len;
                        if (len > 0) {
                            if (!try cb.content(self.bufData()[0..len])) {
                                self.is_valid = false;
                                return false;
                            }
                            self.bufErase(len);
                        }
                        return true;
                    }
                },
                4 => {
                    if (crlf.len > self.bufSize()) return true;
                    if (self.bufStartWith(crlf)) {
                        self.bufErase(crlf.len);
                        self.state = 1;
                    } else if (self.bufStartWith(dash)) {
                        self.bufErase(dash.len);
                        self.is_valid = true;
                        self.state = 5;
                    } else {
                        self.is_valid = false;
                        return false;
                    }
                },
                else => self.bufErase(self.bufSize()),
            }
        }
        return true;
    }
};

// -----------------------------------------------------------------------------
// Ranges

/// Ports `coalesce_ranges` (httplib.cpp:5289 @cd959bae9): ranges sorted by
/// start with libc++'s `std::sort`, normalized, overlapping or adjacent ones
/// merged -- identical ones kept.
fn coalesceRanges(a: std.mem.Allocator, ranges: *std.ArrayList(Range), content_length: usize) error{OutOfMemory}!void {
    if (ranges.items.len <= 1) return;
    const S = struct {
        fn lt(_: void, x: Range, y: Range) bool {
            return x.first < y.first;
        }
    };
    libcxx.sort(Range, ranges.items, {}, S.lt);
    const cl: isize = @intCast(content_length);
    var coalesced: std.ArrayList(Range) = .empty;
    for (ranges.items) |r| {
        var first_pos = r.first;
        var last_pos = r.second;
        if (first_pos == -1 and last_pos == -1) {
            first_pos = 0;
            last_pos = cl;
        }
        if (first_pos == -1) {
            first_pos = cl - last_pos;
            last_pos = cl - 1;
        }
        if (last_pos == -1 or last_pos >= cl) last_pos = cl - 1;
        if (!(0 <= first_pos and first_pos <= last_pos and last_pos < cl)) continue;
        if (coalesced.items.len != 0) {
            const prev = &coalesced.items[coalesced.items.len - 1];
            if (first_pos <= prev.second + 1 and !(first_pos == prev.first and last_pos == prev.second)) {
                prev.second = @max(prev.second, last_pos);
                continue;
            }
        }
        try coalesced.append(a, .{ .first = first_pos, .second = last_pos });
    }
    ranges.* = coalesced;
}

/// Ports `range_error` (httplib.cpp:5345 @cd959bae9): whether a 2xx
/// response's requested ranges cannot be served -- out of bounds, more than
/// `CPPHTTPLIB_RANGE_MAX_COUNT`, or overlapping more than twice. Normalizes
/// and coalesces them otherwise; a streamed body drops them.
pub fn rangeError(req: *Request, res: *Response) error{OutOfMemory}!bool {
    if (req.ranges.items.len != 0 and 200 <= res.status and res.status < 300) {
        if (res.body.items.len == 0 and res.content_provider != null and res.content_length == 0) {
            req.ranges.clearRetainingCapacity();
            if (res.status == detail.StatusCode.partial_content_206) res.status = detail.StatusCode.ok_200;
            return false;
        }
        const content_len: isize = @intCast(if (res.content_length != 0) res.content_length else res.body.items.len);
        var processed: std.ArrayList(Range) = .empty;
        defer processed.deinit(req.a);
        var overwrapping_count: usize = 0;
        if (req.ranges.items.len > detail.range_max_count) return true;
        for (req.ranges.items) |*r| {
            if (r.first == -1 and r.second == -1) {
                r.first = 0;
                r.second = content_len;
            }
            // RFC 9110 14.1.2: a suffix-length longer than the representation
            // selects the entire representation (0.60.0)
            if (r.first == -1) {
                r.first = @max(0, content_len - r.second);
                r.second = content_len - 1;
            }
            if (r.second == -1 or r.second >= content_len) r.second = content_len - 1;
            if (!(0 <= r.first and r.first <= r.second and r.second <= content_len - 1)) return true;
            for (processed.items) |p| {
                if (!(r.second < p.first or r.first > p.second)) {
                    overwrapping_count += 1;
                    if (overwrapping_count > 2) return true;
                    break;
                }
            }
            try processed.append(req.a, r.*);
        }
        try coalesceRanges(req.a, &req.ranges, @intCast(content_len));
    }
    return false;
}

/// Ports `has_faulty_transfer_encoding` (httplib.cpp:3910 @cd959bae9): RFC
/// 9112 6.1: a message framed by Transfer-Encoding next to a Content-Length,
/// or in HTTP/1.0, leaves the connection unfit to carry another one.
pub fn hasFaultyTransferEncoding(msg: anytype) bool {
    return detail.hasHeader(&msg.headers, "Transfer-Encoding") and
        (detail.hasHeader(&msg.headers, "Content-Length") or std.mem.eql(u8, msg.version, "HTTP/1.0"));
}

/// Ports `body_allowed_for_status` (httplib.cpp:7379 @cd959bae9): RFC 9112
/// 6.3: a 1xx, 204 or 304 response ends with its header section.
pub fn bodyAllowedForStatus(status: c_int) bool {
    return status >= 200 and status != 204 and status != 304;
}

/// Ports `set_framing_headers` (httplib.cpp:7395 @cd959bae9): the
/// Content-Length or Transfer-Encoding framing the body `write_response_core`
/// writes, or a closed connection after a body that has neither. Only the
/// library sets them: one from a handler would go out next to it, or describe
/// some other body.
pub fn setFramingHeaders(req: *const Request, res: *Response) error{OutOfMemory}!void {
    const is_head = std.mem.eql(u8, req.method, "HEAD");
    // A HEAD response may give the length a GET would have had (RFC 9110 9.3.2).
    const head_length: []const u8 = if (is_head) res.getHeaderValue("Content-Length") else "";
    _ = res.headers.eraseKey("Content-Length");
    _ = res.headers.eraseKey("Transfer-Encoding");

    if (!bodyAllowedForStatus(res.status)) return;

    if (res.content_provider != null) {
        if (res.content_length == 0) {
            if (res.is_chunked_content_provider) {
                try res.setHeader("Transfer-Encoding", "chunked");
            } else if (!is_head) {
                // RFC 9112 6.3: a body with neither ends where the connection does.
                _ = res.headers.eraseKey("Keep-Alive");
                if (!detail.hasHeaderToken(&res.headers, "Connection", "close")) try res.setHeader("Connection", "close");
            }
            return;
        }
        // the part `write_content_with_provider` writes: every provider here
        // is whole, a known-length one only ever coming from a 200
        try res.setHeader("Content-Length", try std.fmt.allocPrint(res.a, "{d}", .{res.content_length}));
    } else if (res.body.items.len != 0) {
        try res.setHeader("Content-Length", try std.fmt.allocPrint(res.a, "{d}", .{res.body.items.len}));
    } else {
        try res.setHeader("Content-Length", if (head_length.len == 0) "0" else head_length);
    }
}

/// Ports `get_range_offset_and_length` (httplib.cpp:5427 @cd959bae9).
pub fn getRangeOffsetAndLength(r: Range) struct { offset: usize, length: usize } {
    return .{ .offset = @intCast(r.first), .length = @as(usize, @intCast(r.second - r.first)) + 1 };
}

/// Ports `make_content_range_header_field` (httplib.cpp:5437 @cd959bae9).
///
/// Return: allocated by `a`.
pub fn makeContentRangeHeaderField(a: std.mem.Allocator, offset: usize, length: usize, content_length: usize) error{OutOfMemory}![]u8 {
    return std.fmt.allocPrint(a, "bytes {d}-{d}/{d}", .{ offset, offset + length - 1, content_length });
}

/// Ports `make_multipart_data_boundary` (httplib.cpp:5095 @cd959bae9), with
/// `random_string` (httplib.cpp:5083 @cd959bae9): 16 random alphanumerics
/// from `fill_random_bytes`, so no two runs agree on it.
pub fn makeMultipartDataBoundary(a: std.mem.Allocator) error{OutOfMemory}![]u8 {
    const data = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    var out: [16]u8 = undefined;
    @import("ws.zig").fillRandomBytes(&out);
    for (&out) |*c| c.* = data[c.* % data.len];
    return std.mem.concat(a, u8, &.{ "--cpp-httplib-multipart-data-", &out });
}

/// Ports `process_multipart_ranges_data` (httplib.cpp:5452 @cd959bae9) as
/// `make_multipart_ranges_data` (httplib.cpp:5488 @cd959bae9) runs it: each
/// range of `body` as its own part.
///
/// Return: the multipart body, allocated by `a`.
pub fn makeMultipartRangesData(a: std.mem.Allocator, ranges: []const Range, boundary: []const u8, content_type: []const u8, body: []const u8) error{OutOfMemory}![]u8 {
    var data: std.ArrayList(u8) = .empty;
    for (ranges) |r| {
        try data.appendSlice(a, "--");
        try data.appendSlice(a, boundary);
        try data.appendSlice(a, "\r\n");
        if (content_type.len != 0) {
            try data.appendSlice(a, "Content-Type: ");
            try data.appendSlice(a, content_type);
            try data.appendSlice(a, "\r\n");
        }
        const ol = getRangeOffsetAndLength(r);
        try data.appendSlice(a, "Content-Range: ");
        try data.appendSlice(a, try makeContentRangeHeaderField(a, ol.offset, ol.length, body.len));
        try data.appendSlice(a, "\r\n");
        try data.appendSlice(a, "\r\n");
        try data.appendSlice(a, body[ol.offset..][0..ol.length]);
        try data.appendSlice(a, "\r\n");
    }
    try data.appendSlice(a, "--");
    try data.appendSlice(a, boundary);
    try data.appendSlice(a, "--");
    return data.toOwnedSlice(a);
}

// -----------------------------------------------------------------------------
// Writing

/// Ports `write_response_line` (httplib.cpp:4056 @cd959bae9).
pub fn writeResponseLine(strm: anytype, status: c_int) isize {
    var buf: [128]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "HTTP/1.1 {d} {s}\r\n", .{ status, detail.statusMessage(status) }) catch |err| switch (err) {
        error.NoSpaceLeft => unreachable,
    };
    return strm.write(s);
}

/// Ports `write_headers` (httplib.cpp:4065 @cd959bae9): each field as its
/// own write, then the blank line. Since 0.64.0 it writes what it is given:
/// the server drops invalid fields first (`erase_invalid_fields`), and a
/// client checks its own before writing.
pub fn writeHeaders(a: std.mem.Allocator, strm: anytype, headers: *const Headers) error{OutOfMemory}!isize {
    var write_len: isize = 0;
    for (headers.entries.items) |x| {
        const s = try std.mem.concat(a, u8, &.{ x.key, ": ", x.value, "\r\n" });
        defer a.free(s);
        const len = strm.write(s);
        if (len < 0) return len;
        write_len += len;
    }
    const len = strm.write("\r\n");
    if (len < 0) return len;
    write_len += len;
    return write_len;
}

/// Ports `write_data` (httplib.cpp:4084 @cd959bae9): until all of `d` is
/// out.
pub fn writeData(strm: anytype, d: []const u8) bool {
    var offset: usize = 0;
    while (offset < d.len) {
        const length = strm.write(d[offset..]);
        if (length < 0) return false;
        offset += @intCast(length);
    }
    return true;
}

/// Ports `write_content_chunked` (httplib.cpp:4294 @cd959bae9) with the
/// identity coder (`nocompressor::compress`, httplib.cpp:3171 @cd959bae9):
/// calls the provider until it says `done`, each non-empty write as one
/// chunk, then `0\r\n\r\n`. Before each call the peer must still be there.
///
/// Return: false when the peer went away, the provider failed or wrote
/// short, or the server shut down first; an error the provider raised.
pub fn writeContentChunked(a: std.mem.Allocator, strm: *SocketStream, provider: ContentProvider, svr_sock: *const std.atomic.Value(socket.Socket)) anyerror!bool {
    const State = struct {
        a: std.mem.Allocator,
        strm: *SocketStream,
        offset: usize = 0,
        data_available: bool = true,
        ok: bool = true,

        fn write(ctx: *anyopaque, d: []const u8) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (self.ok and d.len > 0) {
                self.offset += d.len;
                var hex_buf: [16]u8 = undefined;
                const hex = detail.fromIToHex(&hex_buf, d.len);
                const chunk = std.mem.concat(self.a, u8, &.{ hex, "\r\n", d, "\r\n" }) catch {
                    self.ok = false;
                    return false;
                };
                defer self.a.free(chunk);
                if (!writeData(self.strm, chunk)) self.ok = false;
            }
            return self.ok;
        }

        fn done(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (!self.ok) return;
            self.data_available = false;
            if (!writeData(self.strm, "0\r\n")) self.ok = false;
            if (!writeData(self.strm, "\r\n")) self.ok = false;
        }
    };
    var state: State = .{ .a = a, .strm = strm };
    var sink: DataSink = .{ .ctx = &state, .write_fn = State.write, .done_fn = State.done };
    while (state.data_available and svr_sock.load(.seq_cst) != socket.invalid_socket) {
        if (!strm.waitWritable() or !strm.isPeerAlive()) {
            return false;
        } else if (!try provider.func(provider.ctx, state.offset, &sink)) {
            return false;
        } else if (!state.ok) {
            return false;
        }
    }
    if (state.data_available) return false;
    return true;
}

/// Ports `write_content_with_progress` (httplib.cpp:4095 @cd959bae9), as
/// `write_content` (httplib.cpp:4181 @cd959bae9) calls it with no progress: a
/// body framed by `length`, written raw. The server reaches it with a
/// chunked provider (through `ContentProviderAdapter`, httplib.cpp:5664
/// @cd959bae9, which drops the length) once its error handler gave a
/// streamed response a body, and so a `Content-Length` (Stage 10 step 6:
/// a 404 through the CORS proxy).
///
/// Return: false when the provider stops, ends early or makes no
/// progress, or a write fails.
pub fn writeContent(strm: *SocketStream, provider: ContentProvider, offset_in: usize, length: usize, svr_sock: *const std.atomic.Value(socket.Socket)) anyerror!bool {
    const State = struct {
        strm: *SocketStream,
        offset: usize,
        end_offset: usize,
        ok: bool = true,
        finished: bool = false,

        fn write(ctx: *anyopaque, d_in: []const u8) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (self.ok) {
                var d = d_in;
                // Bytes past `length` would follow the framed body, where the peer reads
                // them as the next message. Write only what fits and fail the rest.
                if (d.len > self.end_offset - self.offset) {
                    d = d[0 .. self.end_offset - self.offset];
                    self.ok = false;
                }
                if (writeData(self.strm, d)) {
                    self.offset += d.len;
                } else {
                    self.ok = false;
                }
            }
            return self.ok;
        }

        // The body is framed by `length`, so a provider that reports
        // itself done early has truncated it.
        fn done(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.finished = true;
        }
    };
    const end_offset = offset_in + length;
    var state: State = .{ .strm = strm, .offset = offset_in, .end_offset = end_offset };
    var sink: DataSink = .{ .ctx = &state, .write_fn = State.write, .done_fn = State.done };
    while (state.offset < end_offset and !state.finished and svr_sock.load(.seq_cst) != socket.invalid_socket) {
        const last_offset = state.offset;
        if (!strm.waitWritable() or !strm.isPeerAlive()) {
            return false;
        } else if (!try provider.func(provider.ctx, state.offset, &sink)) {
            return false;
        } else if (!state.ok) {
            return false;
        }
        // a provider that makes no progress and does not say it is done
        // would be handed the same offset forever
        if (!state.finished and state.offset == last_offset) return false;
    }
    if (state.offset < end_offset) return false; // done() called early, or is_shutting_down()
    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a header line keeps its name and trims its value" {
    const h = parseHeader("X-Name: \t value \t").?;
    try std.testing.expectEqualStrings("X-Name", h.key);
    try std.testing.expectEqualStrings("value", h.val);
    try std.testing.expect(parseHeader("Bad Name: v") == null);
    try std.testing.expect(parseHeader("NoColon") == null);
}
