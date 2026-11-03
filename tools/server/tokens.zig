//! A prompt as the server holds it -- text tokens with media chunks between
//! them -- and every way a request's `prompt` becomes one.
//!
//! # Provenance
//!
//! Ports `server_tokens` and its state serializer, the tokenizer helpers
//! (`tokenize_mixed`, `tokenize_input_subprompt`, `tokenize_input_prompts`,
//! `process_mtmd_prompt`), the server's `base64_decode`, and
//! `format_prompt_infill` and `format_prompt_rerank` from
//! `llama.cpp/tools/server/server-common.cpp` and
//! `llama.cpp/tools/server/server-common.h` (llama.cpp v0.6.0,
//! `d81235049`). Each declaration names what it ports and the line it
//! began at.
//!
//! # Ownership
//!
//! A `ServerTokens` owns its token list and a copy of every media chunk in
//! it (`mtmd::input_chunk_ptr`), freed by `deinit`. It is move-only, as the
//! C++ is; `clone` is the deep copy.
//!
//! # Process-wide statics
//!
//! `format_prompt_infill` tokenizes three fixed strings into function-local
//! `static`s on its first call, with that call's vocabulary, and reuses them
//! for every later call whatever the vocabulary. The port keeps the same
//! three statics.

const std = @import("std");
const json = @import("json");
const chat = @import("common_chat");
const tools = @import("tools_common");
const common = @import("common.zig");

const c = tools.c;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Error = common.Error;
const Token = common.Token;
const token_null = common.token_null;
const throw = common.throw;
const ggmlAssert = common.ggmlAssert;

/// `raw_buffer` (server-common.h:40 @d81235049): a file's bytes.
pub const RawBuffer = []u8;

/// One entry of `map_idx_to_media`: a media chunk's **start** index in the
/// token list, and the chunk.
pub const Media = struct {
    idx: usize,
    chunk: *c.mtmd_input_chunk,
};

/// Ports `server_tokens` (server-common.h:152 @d81235049): the input tokens,
/// where `LLAMA_TOKEN_NULL` marks a position a media chunk occupies, and the
/// chunks by start index. For M-RoPE a chunk can occupy a different number
/// of positions than tokens.
pub const ServerTokens = struct {
    /// The C allocator by default, so a default-constructed one (as
    /// `server_task()` holds) owns nothing it cannot free.
    gpa: Allocator = std.heap.c_allocator,
    has_mtmd: bool = false,

    // map a **start** index in tokens to the image chunk
    // note: the order need to be in-sync with tokens
    map_idx_to_media: std.ArrayList(Media) = .empty,

    // list of tokens
    //   if the token is LLAMA_TOKEN_NULL, it indicates that this position is occupied by media chunk
    //   otherwise, it is a normal text token
    tokens: std.ArrayList(Token) = .empty,

    /// `server_tokens()`.
    pub fn init(gpa: Allocator) ServerTokens {
        return .{ .gpa = gpa };
    }

    /// The destructor: frees the list and every chunk.
    pub fn deinit(self: *ServerTokens) void {
        for (self.map_idx_to_media.items) |m| c.mtmd_input_chunk_free(m.chunk);
        self.map_idx_to_media.deinit(self.gpa);
        self.tokens.deinit(self.gpa);
    }

    /// Ports the constructor from `mtmd`'s `input_chunks` (server-common.cpp:421 @d81235049).
    pub fn fromChunks(gpa: Allocator, mtmd_chunks: ?*const c.mtmd_input_chunks, has_mtmd: bool) Error!ServerTokens {
        var self: ServerTokens = .{ .gpa = gpa, .has_mtmd = has_mtmd };
        errdefer self.deinit();
        const n = c.mtmd_input_chunks_size(mtmd_chunks);
        for (0..n) |i| {
            try self.pushBackChunk(c.mtmd_input_chunks_get(mtmd_chunks, i));
        }
        return self;
    }

    /// Ports the constructor from `llama_tokens` (server-common.cpp:427 @d81235049).
    pub fn fromTokens(gpa: Allocator, tokens: []const Token, has_mtmd: bool) Error!ServerTokens {
        var self: ServerTokens = .{ .gpa = gpa, .has_mtmd = has_mtmd };
        try self.tokens.appendSlice(gpa, tokens);
        return self;
    }

    /// Ports `server_tokens`'s `operator` `[]` (server-common.h:189 @d81235049), the const one.
    pub fn at(self: *const ServerTokens, index: usize) Token {
        return self.tokens.items[index];
    }

    /// `size` (server-common.h:646 @d81235049).
    pub fn size(self: *const ServerTokens) usize {
        return self.tokens.items.len;
    }

    /// `empty` (server-common.h:622 @d81235049).
    pub fn empty(self: *const ServerTokens) bool {
        return self.tokens.items.len == 0;
    }

    /// `clear` (server-common.h:239 @d81235049).
    pub fn clear(self: *ServerTokens) void {
        for (self.map_idx_to_media.items) |m| c.mtmd_input_chunk_free(m.chunk);
        self.map_idx_to_media.clearRetainingCapacity();
        self.tokens.clearRetainingCapacity();
    }

    fn mediaFind(self: *const ServerTokens, idx: usize) ?usize {
        for (self.map_idx_to_media.items, 0..) |m, i| {
            if (m.idx == idx) return i;
            if (m.idx > idx) return null;
        }
        return null;
    }

    /// `map_idx_to_media[idx] = std::move(chunk)`: inserted where it sorts,
    /// or replacing (and freeing) the chunk already there.
    fn mediaPut(self: *ServerTokens, idx: usize, chunk: *c.mtmd_input_chunk) Error!void {
        var i: usize = 0;
        while (i < self.map_idx_to_media.items.len) : (i += 1) {
            const m = &self.map_idx_to_media.items[i];
            if (m.idx == idx) {
                c.mtmd_input_chunk_free(m.chunk);
                m.chunk = chunk;
                return;
            }
            if (m.idx > idx) break;
        }
        self.map_idx_to_media.insert(self.gpa, i, .{ .idx = idx, .chunk = chunk }) catch |e| {
            c.mtmd_input_chunk_free(chunk);
            return e;
        };
    }

    /// Ports `pos_next` (server-common.cpp:430 @d81235049): the next position
    /// after `n_tokens` tokens, or after all of them when `n_tokens < 0`.
    pub fn posNext(self: *const ServerTokens, n_tokens: i64) c.llama_pos {
        if (!self.has_mtmd) {
            if (n_tokens < 0) {
                return @truncate(@as(i64, @intCast(self.tokens.items.len)));
            }

            return @truncate(n_tokens);
        }

        if (n_tokens < 0) {
            var res: i64 = @intCast(self.tokens.items.len);
            for (self.map_idx_to_media.items) |m| {
                // `int32_t - size_t`: unsigned, then back to `llama_pos`
                res +%= @as(i64, c.mtmd_input_chunk_get_n_pos(m.chunk)) -% @as(i64, @intCast(c.mtmd_input_chunk_get_n_tokens(m.chunk)));
            }

            return @truncate(res);
        }

        var idx: i64 = 0;
        var pos: c.llama_pos = 0;

        ggmlAssert(n_tokens <= @as(i64, @intCast(self.tokens.items.len)), "server-common.cpp", 453, "n_tokens <= (int64_t)tokens.size()");

        while (idx < n_tokens) {
            if (self.mediaFind(@intCast(idx))) |mi| {
                const chunk = self.map_idx_to_media.items[mi].chunk;
                const n_pos = c.mtmd_input_chunk_get_n_pos(chunk);
                const n_tok = c.mtmd_input_chunk_get_n_tokens(chunk);

                pos +%= n_pos;
                idx += @intCast(n_tok);
            } else {
                pos +%= 1;
                idx += 1;
            }
        }

        return pos;
    }

    /// Ports `size_up_to_pos` (server-common.cpp:473 @d81235049): the number
    /// of tokens with a position below `max_pos`.
    pub fn sizeUpToPos(self: *const ServerTokens, max_pos: c.llama_pos) usize {
        if (!self.has_mtmd) {
            const mp: usize = @bitCast(@as(isize, max_pos));
            return @min(mp, self.tokens.items.len);
        }

        var idx: usize = 0;
        var pos: c.llama_pos = 0;

        while (idx < self.tokens.items.len) {
            if (self.mediaFind(idx)) |mi| {
                const chunk = self.map_idx_to_media.items[mi].chunk;
                const n_pos = c.mtmd_input_chunk_get_n_pos(chunk);
                const n_tok = c.mtmd_input_chunk_get_n_tokens(chunk);

                pos +%= n_pos;
                idx += n_tok;
            } else {
                pos +%= 1;
                idx += 1;
            }

            if (pos >= max_pos) {
                break;
            }
        }

        return idx;
    }

    /// Ports `str` (server-common.cpp:503 @d81235049), for debugging.
    ///
    /// Return: owned by `a`.
    pub fn str(self: *const ServerTokens, a: Allocator) Error![]u8 {
        var oss: std.ArrayList(u8) = .empty;
        errdefer oss.deinit(a);
        try oss.appendSlice(a, "tokens: ");
        for (self.tokens.items, 0..) |t, idx| {
            try oss.print(a, "idx:{d} ", .{idx});
            if (t == token_null) {
                try oss.appendSlice(a, "<embd> ");
            } else {
                try oss.print(a, "{d} ", .{t});
            }
        }
        try oss.appendSlice(a, "\n");
        try oss.appendSlice(a, "image idx: ");
        for (self.map_idx_to_media.items) |m| {
            try oss.print(a, "{d}, ", .{m.idx});
        }
        return oss.toOwnedSlice(a);
    }

    /// Ports `find_chunk` (server-common.cpp:523 @d81235049).
    ///
    /// Return: the chunk starting at `idx`, borrowed.
    pub fn findChunk(self: *const ServerTokens, idx: usize) Error!*c.mtmd_input_chunk {
        if (self.mediaFind(idx)) |mi| {
            return self.map_idx_to_media.items[mi].chunk;
        }
        return throw(error.RuntimeError, "Chunk not found", .{});
    }

    /// Ports `find_next_media_chunk` (server-common.cpp:531 @d81235049): the
    /// first media chunk starting after `idx`.
    pub fn findNextMediaChunk(self: *const ServerTokens, idx: usize) ?Media {
        for (self.map_idx_to_media.items) |m| {
            if (m.idx > idx) return m;
        }
        return null;
    }

    /// Ports the `push_back` overload taking a token (server-common.cpp:539 @d81235049).
    pub fn pushBackToken(self: *ServerTokens, tok: Token) Error!void {
        if (tok == token_null) {
            return throw(error.RuntimeError, "Invalid token", .{});
        }
        try self.tokens.append(self.gpa, tok);
    }

    /// Ports the `push_back` overload taking a chunk (server-common.cpp:546 @d81235049):
    /// a media chunk is copied, a text chunk's tokens appended.
    pub fn pushBackChunk(self: *ServerTokens, chunk: ?*const c.mtmd_input_chunk) Error!void {
        const ty = c.mtmd_input_chunk_get_type(chunk);
        if (ty == c.MTMD_INPUT_CHUNK_TYPE_IMAGE or ty == c.MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            ggmlAssert(self.has_mtmd, "server-common.cpp", 549, "has_mtmd");
            const n_tokens = c.mtmd_input_chunk_get_n_tokens(chunk);
            const start_idx = self.tokens.items.len;
            try self.tokens.appendNTimes(self.gpa, token_null, n_tokens);
            const new_chunk = c.mtmd_input_chunk_copy(chunk) orelse unreachable;
            try self.mediaPut(start_idx, new_chunk);
        } else if (ty == c.MTMD_INPUT_CHUNK_TYPE_TEXT) {
            var n_tokens: usize = 0;
            const text_tokens = c.mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            for (0..n_tokens) |i| {
                try self.pushBackToken(text_tokens[i]);
            }
        } else {
            _ = c.ggml_abort("server-common.cpp", 564, "Invalid chunk type");
        }
    }

    /// Ports `push_back_placeholder` (server-common.cpp:568 @d81235049): as
    /// `pushBackChunk`, with a media chunk stored as a placeholder.
    pub fn pushBackPlaceholder(self: *ServerTokens, chunk: ?*const c.mtmd_input_chunk) Error!void {
        const ty = c.mtmd_input_chunk_get_type(chunk);
        if (ty == c.MTMD_INPUT_CHUNK_TYPE_IMAGE or ty == c.MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            ggmlAssert(self.has_mtmd, "server-common.cpp", 571, "has_mtmd");
            const new_chunk = c.mtmd_input_chunk_get_placeholder(chunk);
            ggmlAssert(new_chunk != null, "server-common.cpp", 573, "new_chunk != nullptr && \"failed to create placeholder chunk\"");
            const n_tokens = c.mtmd_input_chunk_get_n_tokens(chunk);
            const start_idx = self.tokens.items.len;
            self.tokens.appendNTimes(self.gpa, token_null, n_tokens) catch |e| {
                c.mtmd_input_chunk_free(new_chunk);
                return e;
            };
            try self.mediaPut(start_idx, new_chunk.?);
        } else {
            try self.pushBackChunk(chunk);
        }
    }

    /// Ports the `push_back` overload taking `server_tokens` (server-common.cpp:427 @d81235049):
    /// appends `other`'s tokens and copies its chunks. Every media position
    /// holds `LLAMA_TOKEN_NULL`, which the token `push_back` refuses, so
    /// with media this throws `Invalid token` before reaching the chunk loop
    /// -- whose iterator, as upstream writes it, never advances.
    pub fn pushBackTokens(self: *ServerTokens, other: *const ServerTokens) Error!void {
        const start_idx = self.size();
        for (0..other.size()) |i| {
            try self.pushBackToken(other.at(i));
        }
        if (other.has_mtmd) {
            // Assert if we are copying MTMD chunks to a server_tokens that does not have mtmd.
            // We could also just check, but this will prevent silently dropping MTMD data.
            ggmlAssert(self.has_mtmd, "server-common.cpp", 593, "has_mtmd");
            const items = other.map_idx_to_media.items;
            // the C++ loop's iterator is never incremented
            const it: usize = 0;
            while (it < items.len) {
                const chunk = items[it].chunk;
                const new_chunk = c.mtmd_input_chunk_copy(chunk) orelse unreachable;
                try self.mediaPut(start_idx + items[it].idx, new_chunk);
            }
        }
    }

    /// Ports `insert` (server-common.cpp:602 @d81235049): for compatibility
    /// with context shift and prompt truncation.
    pub fn insert(self: *ServerTokens, inp_tokens: []const Token) Error!void {
        try self.tokens.appendSlice(self.gpa, inp_tokens);
    }

    /// Ports `get_tokens` (server-common.cpp:606 @d81235049).
    ///
    /// Return: borrowed until the next change.
    pub fn getTokens(self: *const ServerTokens) []const Token {
        ggmlAssert(!self.has_mtmd, "server-common.cpp", 607, "!has_mtmd");
        return self.tokens.items;
    }

    /// Ports `get_text_tokens` (server-common.cpp:678 @d81235049).
    ///
    /// Return: owned by `a`.
    pub fn getTextTokens(self: *const ServerTokens, a: Allocator) Error![]Token {
        var res: std.ArrayList(Token) = .empty;
        errdefer res.deinit(a);
        try res.ensureTotalCapacity(a, self.tokens.items.len);
        for (self.tokens.items) |t| {
            if (t != token_null) {
                res.appendAssumeCapacity(t);
            }
        }
        return res.toOwnedSlice(a);
    }

    /// Ports `set_token` (server-common.cpp:689 @d81235049).
    pub fn setToken(self: *ServerTokens, pos: c.llama_pos, id: Token) void {
        ggmlAssert(!self.has_mtmd, "server-common.cpp", 690, "!has_mtmd"); // only allow this if mtmd is disabled
        self.tokens.items[@intCast(pos)] = id;
    }

    /// Ports `keep_first` (server-common.cpp:694 @d81235049): truncates to
    /// `n` tokens, refusing to cut through a media chunk.
    pub fn keepFirst(self: *ServerTokens, n: usize) Error!void {
        ggmlAssert(n <= self.tokens.items.len, "server-common.cpp", 695, "n <= tokens.size()");
        if (self.has_mtmd) {
            if (n == self.tokens.items.len) {
                return; // nothing to do
            }
            // we throw an error if we try to remove a token in the middle of an image
            if (n > 0) {
                // make sure we never remove tokens in the middle of an image
                // note that the case where we keep a full image at the end is allowed:
                //   tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] != LLAMA_TOKEN_NULL
                if (self.tokens.items[n - 1] == token_null and self.tokens.items[n] == token_null) {
                    _ = try self.findChunk(n); // will throw an error if the cut is not at a chunk boundary
                }
            }
            // remove all image chunks that are not used anymore
            var i: usize = 0;
            while (i < self.map_idx_to_media.items.len) {
                const m = self.map_idx_to_media.items[i];
                if (m.idx >= n) {
                    c.mtmd_input_chunk_free(m.chunk);
                    _ = self.map_idx_to_media.orderedRemove(i);
                } else {
                    i += 1;
                }
            }
        }
        self.tokens.shrinkRetainingCapacity(n);
    }

    /// Ports `detokenize` (server-common.cpp:727 @d81235049): the text tokens'
    /// text.
    ///
    /// Return: owned by `a`.
    pub fn detokenize(self: *const ServerTokens, a: Allocator, ctx: ?*const c.llama_context, special: bool) Error![]u8 {
        const text_tokens = try self.getTextTokens(a);
        defer a.free(text_tokens);
        const model = c.llama_get_model(ctx);
        return common.detokenize(a, c.llama_model_get_vocab(model), text_tokens, special);
    }

    /// Ports `get_common_prefix` (server-common.cpp:738 @d81235049): equal
    /// media chunks (same id, same length) count as equal tokens.
    pub fn getCommonPrefix(self: *const ServerTokens, b: *const ServerTokens) Error!usize {
        const max_idx = @min(self.tokens.items.len, b.tokens.items.len);

        if (!self.has_mtmd) {
            for (0..max_idx) |i| {
                if (self.tokens.items[i] == b.tokens.items[i]) {
                    continue;
                }

                return i;
            }

            return max_idx;
        }

        var i: usize = 0;
        while (i < max_idx) : (i += 1) {
            const ai = self.tokens.items[i];
            const bi = b.tokens.items[i];

            if (ai == token_null and bi == token_null) {
                const a_chunk = try self.findChunk(i);
                const b_chunk = try b.findChunk(i);

                const id_ai = std.mem.span(@as([*:0]const u8, @ptrCast(c.mtmd_input_chunk_get_id(a_chunk))));
                const id_bi = std.mem.span(@as([*:0]const u8, @ptrCast(c.mtmd_input_chunk_get_id(b_chunk))));

                const n_tok_a = c.mtmd_input_chunk_get_n_tokens(a_chunk);
                const n_tok_b = c.mtmd_input_chunk_get_n_tokens(b_chunk);

                if (std.mem.eql(u8, id_ai, id_bi) and n_tok_a == n_tok_b) {
                    ggmlAssert(n_tok_a > 0, "server-common.cpp", 770, "n_tok_a > 0 && \"Invalid media chunk\""); // should never happen
                    i += n_tok_a - 1; // will be +1 by the for loop
                    continue;
                }

                return i;
            }

            if (ai == bi) {
                continue;
            }

            return i;
        }

        return max_idx; // all tokens are equal
    }

    /// Ports `find_message_spans` (server-common.cpp:788 @d81235049): the
    /// tokens split into message spans, skipping over media chunks.
    ///
    /// Return: owned by `a`.
    pub fn findMessageSpans(self: *const ServerTokens, a: Allocator, delims: *const chat.MsgDelimiters) Error!chat.MsgSpans {
        const skips = try a.alloc([2]usize, self.map_idx_to_media.items.len);
        defer a.free(skips);
        for (self.map_idx_to_media.items, 0..) |m, i| {
            skips[i] = .{ m.idx, c.mtmd_input_chunk_get_n_tokens(m.chunk) };
        }
        return delims.split(a, self.tokens.items, skips) catch |e| switch (e) {
            error.OutOfMemory => error.OutOfMemory,
        };
    }

    /// Ports `validate` (server-common.cpp:796 @d81235049): every text token in
    /// the vocabulary, every media position covered by exactly one chunk.
    pub fn validate(self: *const ServerTokens, ctx: ?*const c.llama_context) bool {
        const model = c.llama_get_model(ctx);
        const vocab = c.llama_model_get_vocab(model);
        const n_vocab = c.llama_vocab_n_tokens(vocab);
        var n_media: usize = 0;

        var i: usize = 0;
        while (i < self.tokens.items.len) : (i += 1) {
            const t = self.tokens.items[i];
            if (t == token_null) {
                const chunk = self.findChunk(i) catch return false;
                if (c.mtmd_input_chunk_get_type(chunk) == c.MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    return false;
                }
                const n_tokens = c.mtmd_input_chunk_get_n_tokens(chunk);
                const n_pos = c.mtmd_input_chunk_get_n_pos(chunk);
                if (n_tokens == 0 or n_pos <= 0 or n_tokens > self.tokens.items.len - i) {
                    return false;
                }
                for (i..i + n_tokens) |j| {
                    if (self.tokens.items[j] != token_null) {
                        return false;
                    }
                }
                n_media += 1;
                i += n_tokens - 1;
            } else if (t < 0 or t >= n_vocab) {
                return false;
            }
        }
        return n_media == self.map_idx_to_media.items.len;
    }

    /// Ports `clone` (server-common.cpp:832 @d81235049): a deep copy, chunks
    /// included.
    pub fn clone(self: *const ServerTokens) Error!ServerTokens {
        var res: ServerTokens = .{ .gpa = self.gpa, .has_mtmd = self.has_mtmd };
        errdefer res.deinit();
        try res.tokens.appendSlice(self.gpa, self.tokens.items);
        for (self.map_idx_to_media.items) |m| {
            try res.mediaPut(m.idx, c.mtmd_input_chunk_copy(m.chunk) orelse unreachable);
        }
        return res;
    }

    // -------------------------------------------------------------------------
    // the state format

    /// Ports `SERVER_TOKENS_STATE_VERSION` (server-common.cpp:327 @d81235049).
    const state_version: u32 = 1;

    /// Ports `server_tokens_state_u32` (server-common.cpp:329 @d81235049).
    fn stateU32(value: usize) Error!u32 {
        if (value > std.math.maxInt(u32)) {
            return throw(error.RuntimeError, "Server tokens state is too large", .{});
        }
        return @intCast(value);
    }

    /// Ports `server_tokens_state_writer` (server-common.cpp:336 @d81235049).
    const Writer = struct {
        a: Allocator,
        data: std.ArrayList(u8) = .empty,

        fn writeU32(self: *Writer, v: u32) Error!void {
            try self.data.appendSlice(self.a, std.mem.asBytes(&v));
        }

        fn writeI32(self: *Writer, v: i32) Error!void {
            try self.data.appendSlice(self.a, std.mem.asBytes(&v));
        }

        /// The `std::vector<T>` `write`: a `u32` count, then the elements' bytes.
        fn writeVector(self: *Writer, bytes: []const u8, n: usize) Error!void {
            try self.writeU32(try stateU32(n));
            if (n == 0) {
                return;
            }
            try self.data.appendSlice(self.a, bytes);
        }

        /// Ports `write_media_chunk` (server-common.cpp:356 @d81235049).
        fn writeMediaChunk(self: *Writer, chunk: *const c.mtmd_input_chunk) Error!void {
            var chunk_size: usize = 0;
            if (c.mtmd_input_chunk_save(chunk, null, 0, &chunk_size) != 0 or chunk_size == 0) {
                return throw(error.RuntimeError, "Cannot serialize media chunk in server tokens", .{});
            }
            const chunk_data = try self.a.alloc(u8, try stateU32(chunk_size));
            defer self.a.free(chunk_data);
            @memset(chunk_data, 0);
            if (c.mtmd_input_chunk_save(chunk, chunk_data.ptr, chunk_data.len, null) != 0) {
                return throw(error.RuntimeError, "Cannot serialize media chunk in server tokens", .{});
            }
            try self.writeVector(chunk_data, chunk_data.len);
        }

        /// Ports `take` (server-common.cpp:368 @d81235049): padded with zeros
        /// to a whole number of tokens.
        fn take(self: *Writer) Error![]u8 {
            const n = (self.data.items.len + @sizeOf(Token) - 1) / @sizeOf(Token) * @sizeOf(Token);
            try self.data.appendNTimes(self.a, 0, n - self.data.items.len);
            return self.data.toOwnedSlice(self.a);
        }
    };

    /// Ports `server_tokens_state_reader` (server-common.cpp:377 @d81235049).
    const Reader = struct {
        data: []const u8,
        pos: usize = 0,

        /// Ports `read` (server-common.cpp:382 @d81235049).
        fn read(self: *Reader, comptime T: type) Error!T {
            if (self.data.len - self.pos < @sizeOf(T)) {
                return throw(error.RuntimeError, "Unexpected end of server tokens state", .{});
            }
            const v = std.mem.bytesToValue(T, self.data[self.pos..][0..@sizeOf(T)]);
            self.pos += @sizeOf(T);
            return v;
        }

        /// Ports `read_vector` (server-common.cpp:394 @d81235049).
        ///
        /// Return: the elements' bytes, borrowed from the state.
        fn readVector(self: *Reader, comptime T: type) Error![]const u8 {
            const n_values = try self.read(u32);
            // reject before resizing, so that a small corrupted payload cannot request a huge allocation
            if (n_values > self.remaining() / @sizeOf(T)) {
                return throw(error.RuntimeError, "Unexpected end of server tokens state", .{});
            }
            const bytes = self.data[self.pos..][0 .. @as(usize, n_values) * @sizeOf(T)];
            self.pos += bytes.len;
            return bytes;
        }

        /// Ports `remaining` (server-common.cpp:409 @d81235049).
        fn remaining(self: *const Reader) usize {
            return self.data.len - self.pos;
        }
    };

    /// Ports `serialize` (server-common.cpp:611 @d81235049): a format marker,
    /// the version, the tokens, the media start indices and each chunk's
    /// saved bytes, padded to a whole number of tokens.
    ///
    /// Return: owned by `a`.
    pub fn serialize(self: *const ServerTokens, a: Allocator) Error![]u8 {
        var writer: Writer = .{ .a = a };
        errdefer writer.data.deinit(a);
        try writer.writeI32(token_null);
        try writer.writeU32(state_version);
        try writer.writeVector(std.mem.sliceAsBytes(self.tokens.items), self.tokens.items.len);

        const media_keys = try a.alloc(u32, self.map_idx_to_media.items.len);
        defer a.free(media_keys);
        for (self.map_idx_to_media.items, 0..) |m, i| {
            media_keys[i] = try stateU32(m.idx);
        }
        try writer.writeVector(std.mem.sliceAsBytes(media_keys), media_keys.len);

        for (self.map_idx_to_media.items) |m| {
            try writer.writeMediaChunk(m.chunk);
        }

        return writer.take();
    }

    /// Ports `deserialize` (server-common.cpp:633 @d81235049): a list that
    /// does not start with the marker is a plain token list, as older
    /// versions wrote.
    pub fn deserialize(gpa: Allocator, packed_tokens: []const Token, has_mtmd: bool) Error!ServerTokens {
        if (packed_tokens.len == 0 or packed_tokens[0] != token_null) {
            // plain token list, as written by older versions
            return fromTokens(gpa, packed_tokens, has_mtmd);
        }

        var reader: Reader = .{ .data = std.mem.sliceAsBytes(packed_tokens) };
        _ = try reader.read(Token); // format marker
        if (try reader.read(u32) != state_version) {
            return throw(error.RuntimeError, "Unsupported server tokens state version", .{});
        }

        const tokens_bytes = try reader.readVector(Token);

        // the media start indices, followed by the media chunks in the same order
        const media_bytes = try reader.readVector(u32);
        if (media_bytes.len != 0 and !has_mtmd) {
            return throw(error.RuntimeError, "Cannot restore media tokens without an mmproj", .{});
        }

        var result: ServerTokens = .{ .gpa = gpa, .has_mtmd = has_mtmd };
        errdefer result.deinit();
        const n_tokens = tokens_bytes.len / @sizeOf(Token);
        try result.tokens.ensureTotalCapacity(gpa, n_tokens);
        for (0..n_tokens) |i| {
            result.tokens.appendAssumeCapacity(std.mem.bytesToValue(Token, tokens_bytes[i * 4 ..][0..4]));
        }

        for (0..media_bytes.len / 4) |k| {
            const start_idx: usize = std.mem.bytesToValue(u32, media_bytes[k * 4 ..][0..4]);
            const chunk_data = try reader.readVector(u8);
            if (chunk_data.len == 0) {
                return throw(error.RuntimeError, "Cannot load media chunk from server tokens state", .{});
            }

            const chunk = c.mtmd_input_chunk_load(chunk_data.ptr, chunk_data.len) orelse
                return throw(error.RuntimeError, "Cannot load media chunk from server tokens state", .{});
            try result.mediaPut(start_idx, chunk);
        }

        if (reader.remaining() >= @sizeOf(Token)) {
            return throw(error.RuntimeError, "Trailing data in server tokens state", .{});
        }

        return result;
    }
};

// -----------------------------------------------------------------------------
// tokenizer and input processing utils

/// Ports `tokenize_mixed` (server-common.cpp:908 @d81235049): a string, or an
/// array mixing strings and token ids. Only the first element, when it is a
/// string, gets the special tokens.
///
/// Return: owned by `a`.
pub fn tokenizeMixed(a: Allocator, vocab: ?*const c.llama_vocab, json_prompt: *const Value, add_special: bool, parse_special: bool) Error![]Token {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    var prompt_tokens: std.ArrayList(Token) = .empty;
    errdefer prompt_tokens.deinit(a);

    if (json_prompt.isArray()) {
        var first = true;
        for (json_prompt.array.items) |*p| {
            if (p.isString()) {
                const s = try p.get([]const u8);

                var toks: []Token = undefined;
                if (first) {
                    toks = try common.tokenize(a, vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    toks = try common.tokenize(a, vocab, s, false, parse_special);
                }
                defer a.free(toks);

                try prompt_tokens.appendSlice(a, toks);
            } else {
                if (first) {
                    first = false;
                }

                try prompt_tokens.append(a, try p.get(Token));
            }
        }
    } else {
        const s = try json_prompt.get([]const u8);
        const toks = try common.tokenize(a, vocab, s, add_special, parse_special);
        defer a.free(toks);
        try prompt_tokens.appendSlice(a, toks);
    }

    return prompt_tokens.toOwnedSlice(a);
}

// base64 utils (TODO: use the base64::decode from base64.hpp)

/// Ports `base64_chars` (server-common.cpp:260 @d81235049).
const base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" ++ "abcdefghijklmnopqrstuvwxyz" ++ "0123456789+/";

/// Ports `is_base64` (server-common.cpp:265 @d81235049): `isalnum` in the
/// "C" locale.
fn isBase64(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or (ch == '+') or (ch == '/');
}

/// `base64_chars.find(ch)`, narrowed to `uint8_t` as the C++ stores it:
/// `npos` becomes 0xFF.
fn findChar(ch: u8) u8 {
    return if (std.mem.indexOfScalar(u8, base64_chars, ch)) |i| @intCast(i) else 0xFF;
}

/// Ports `base64_decode` (server-common.cpp:269 @d81235049): decodes up to the
/// first `=` or the first byte outside the alphabet.
///
/// Return: owned by `a`.
pub fn base64Decode(a: Allocator, encoded_string: []const u8) Error![]u8 {
    var i: usize = 0;
    var in_: usize = 0;

    var in_len = encoded_string.len;

    var char_array_4: [4]u8 = undefined;
    var char_array_3: [3]u8 = undefined;

    var ret: std.ArrayList(u8) = .empty;
    errdefer ret.deinit(a);

    while (in_len != 0) {
        in_len -= 1;
        if (encoded_string[in_] == '=' or !isBase64(encoded_string[in_])) break;
        char_array_4[i] = encoded_string[in_];
        i += 1;
        in_ += 1;
        if (i == 4) {
            for (&char_array_4) |*ch| ch.* = findChar(ch.*);

            char_array_3[0] = (char_array_4[0] << 2) +% ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) +% ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +% char_array_4[3];

            try ret.appendSlice(a, &char_array_3);

            i = 0;
        }
    }

    if (i != 0) {
        for (i..4) |j| char_array_4[j] = 0;

        for (&char_array_4) |*ch| ch.* = findChar(ch.*);

        char_array_3[0] = (char_array_4[0] << 2) +% ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) +% ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +% char_array_4[3];

        try ret.appendSlice(a, char_array_3[0 .. i - 1]);
    }

    return ret.toOwnedSlice(a);
}

/// Ports `process_mtmd_prompt` (server-common.cpp:971 @d81235049): the prompt
/// and its files through `mtmd_tokenize`. With `is_placeholder` the media
/// chunks only count tokens; they cannot be decoded.
pub fn processMtmdPrompt(
    gpa: Allocator,
    mctx: ?*c.mtmd_context,
    prompt: []const u8,
    files: []const []const u8,
    init_opt: c.mtmd_helper_init_opt,
    is_placeholder: bool,
) Error!ServerTokens {
    // these will be freed upon going out of scope
    var bitmaps: std.ArrayList(?*const c.mtmd_bitmap) = .empty;
    defer {
        for (bitmaps.items) |b| c.mtmd_bitmap_free(@constCast(b));
        bitmaps.deinit(gpa);
    }
    var videos: std.ArrayList(*c.mtmd_helper_video) = .empty;
    defer {
        for (videos.items) |v| c.mtmd_helper_video_free(v);
        videos.deinit(gpa);
    }
    for (files) |file| {
        const out = c.mtmd_helper_bitmap_init_from_buf(mctx, file.ptr, file.len, is_placeholder, init_opt);
        if (out.bitmap == null) {
            return throw(error.RuntimeError, "Failed to load image or audio file", .{});
        }
        bitmaps.append(gpa, out.bitmap) catch |e| {
            c.mtmd_bitmap_free(out.bitmap);
            if (out.video_ctx) |v| c.mtmd_helper_video_free(v);
            return e;
        };
        if (out.video_ctx) |v| {
            videos.append(gpa, v) catch |e| {
                c.mtmd_helper_video_free(v);
                return e;
            };
        }
    }
    // process prompt
    // multimodal
    const inp_txt: c.mtmd_input_text = .{
        .text = prompt.ptr,
        .text_len = prompt.len,
        .add_special = true,
        .parse_special = true,
    };
    const chunks = c.mtmd_input_chunks_init();
    defer c.mtmd_input_chunks_free(chunks);
    const tokenized = c.mtmd_tokenize(mctx, chunks, &inp_txt, bitmaps.items.ptr, bitmaps.items.len);
    if (tokenized != 0) {
        return throw(error.RuntimeError, "Failed to tokenize prompt", .{});
    }
    return ServerTokens.fromChunks(gpa, chunks, true);
}

/// Ports `tokenize_input_subprompt` (server-common.cpp:1022 @d81235049): one
/// prompt -- a string, a list of tokens, strings and tokens mixed, or an
/// object with `prompt_string` and optionally `multimodal_data`.
pub fn tokenizeInputSubprompt(
    gpa: Allocator,
    vocab: ?*const c.llama_vocab,
    mctx: ?*c.mtmd_context,
    json_prompt: *const Value,
    add_special: bool,
    parse_special: bool,
    init_opt: c.mtmd_helper_init_opt,
) Error!ServerTokens {
    const json_string_prompt_key = "prompt_string";
    const json_mtmd_data_key = "multimodal_data";
    const has_mtmd = mctx != null;
    if (json_prompt.isString() or common.jsonIsArrayOfMixedNumbersStrings(json_prompt)) {
        // string or mixed
        const tmp = try tokenizeMixed(gpa, vocab, json_prompt, add_special, parse_special);
        defer gpa.free(tmp);
        return ServerTokens.fromTokens(gpa, tmp, false);
    } else if (common.jsonIsArrayOfNumbers(json_prompt)) {
        // array of tokens
        const tmp = try json_prompt.getSlice(gpa, Token);
        defer gpa.free(tmp);
        return ServerTokens.fromTokens(gpa, tmp, false);
    } else if (json_prompt.contains(json_string_prompt_key)) {
        // JSON object with prompt key.
        if (json_prompt.contains(json_mtmd_data_key)) {
            if (!has_mtmd)
                return throw(error.RuntimeError, "Multimodal data provided, but model does not support multimodal requests.", .{});

            // JSON object with prompt and multimodal key.
            var files: std.ArrayList([]const u8) = .empty;
            defer {
                for (files.items) |f| gpa.free(f);
                files.deinit(gpa);
            }
            const data = try json_prompt.at(json_mtmd_data_key);
            var it = data.items();
            while (it.next()) |entry| {
                const decoded = try base64Decode(gpa, try entry.value.get([]const u8));
                files.append(gpa, decoded) catch |e| {
                    gpa.free(decoded);
                    return e;
                };
            }
            const prompt = try (try json_prompt.at(json_string_prompt_key)).get([]const u8);
            return processMtmdPrompt(gpa, mctx, prompt, files.items, init_opt, false);
        } else {
            // Not multimodal, but contains a subobject.
            const tmp = try tokenizeMixed(gpa, vocab, try json_prompt.at(json_string_prompt_key), add_special, parse_special);
            defer gpa.free(tmp);
            return ServerTokens.fromTokens(gpa, tmp, false);
        }
    } else {
        return throw(error.InvalidArgument, "\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.", .{});
    }
}

/// Ports `tokenize_input_prompts` (server-common.cpp:1056 @d81235049): a
/// `prompt` that may hold several prompts (an array without bare tokens),
/// each through `tokenize_input_subprompt`.
///
/// Return: owned by `gpa`; `deinit` each.
pub fn tokenizeInputPrompts(
    gpa: Allocator,
    vocab: ?*const c.llama_vocab,
    mctx: ?*c.mtmd_context,
    json_prompt: *const Value,
    add_special: bool,
    parse_special: bool,
    init_opt: c.mtmd_helper_init_opt,
) Error!std.ArrayList(ServerTokens) {
    var result: std.ArrayList(ServerTokens) = .empty;
    errdefer {
        for (result.items) |*t| t.deinit();
        result.deinit(gpa);
    }
    if (json_prompt.isArray() and !common.jsonIsArrayAndContainsNumbers(json_prompt)) {
        try result.ensureTotalCapacity(gpa, json_prompt.array.items.len);
        for (json_prompt.array.items) |*p| {
            result.appendAssumeCapacity(try tokenizeInputSubprompt(gpa, vocab, mctx, p, add_special, parse_special, init_opt));
        }
    } else {
        var t = try tokenizeInputSubprompt(gpa, vocab, mctx, json_prompt, add_special, parse_special, init_opt);
        result.append(gpa, t) catch |e| {
            t.deinit();
            return e;
        };
    }
    if (result.items.len == 0) {
        return throw(error.InvalidArgument, "\"prompt\" must not be empty", .{});
    }
    return result;
}

// -----------------------------------------------------------------------------
// formatting prompts

/// The function-local `static`s of `format_prompt_infill`: tokenized once,
/// with the first call's vocabulary.
const InfillStatics = struct {
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    k_fim_repo: ?[]Token = null,
    k_chunk_prefix_tokens: ?[]Token = null,
    k_fim_file: ?[]Token = null,

    fn get(self: *InfillStatics, slot: *?[]Token, vocab: ?*const c.llama_vocab, text: []const u8) Error![]const Token {
        _ = std.c.pthread_mutex_lock(&self.mutex);
        defer _ = std.c.pthread_mutex_unlock(&self.mutex);
        if (slot.* == null) slot.* = try common.tokenize(std.heap.c_allocator, vocab, text, false, false);
        return slot.*.?;
    }
};
var infill_statics: InfillStatics = .{};

/// Ports `format_prompt_infill` (server-common.cpp:1778 @d81235049): the
/// FIM repo-level pattern -- extra chunks, then prefix, suffix and the
/// middle marker -- cut to fit a batch.
///
/// Return: owned by `a`.
pub fn formatPromptInfill(
    a: Allocator,
    vocab: ?*const c.llama_vocab,
    input_prefix: *const Value,
    input_suffix: *const Value,
    input_extra: *Value,
    n_batch: c_int,
    n_predict: c_int,
    n_ctx: c_int,
    spm_infill: bool,
    tokens_prompt: []const Token,
) Error![]Token {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    var extra_tokens: std.ArrayList(Token) = .empty;
    defer extra_tokens.deinit(a);
    try extra_tokens.ensureTotalCapacity(a, @intCast(n_ctx));

    var tokens_prefix: std.ArrayList(Token) = .fromOwnedSlice(try tokenizeMixed(a, vocab, input_prefix, false, false));
    defer tokens_prefix.deinit(a);
    var tokens_suffix: std.ArrayList(Token) = .fromOwnedSlice(try tokenizeMixed(a, vocab, input_suffix, false, false));
    defer tokens_suffix.deinit(a);

    if (c.llama_vocab_fim_rep(vocab) != token_null) {
        // TODO: make project name an input
        const k_fim_repo = try infill_statics.get(&infill_statics.k_fim_repo, vocab, "myproject\n");

        try extra_tokens.append(a, c.llama_vocab_fim_rep(vocab));
        try extra_tokens.appendSlice(a, k_fim_repo);
    }
    var it = input_extra.items();
    while (it.next()) |entry| {
        const chunk = entry.value;
        // { "text": string, "filename": string }
        const text = common.jsonValue(chunk, "text", []const u8, "");
        const filename = common.jsonValue(chunk, "filename", []const u8, "tmp");

        if (c.llama_vocab_fim_sep(vocab) != token_null) {
            const name = try std.mem.concat(a, u8, &.{ filename, "\n" });
            defer a.free(name);
            const k_fim_file = try common.tokenize(a, vocab, name, false, false);
            defer a.free(k_fim_file);

            try extra_tokens.append(a, c.llama_vocab_fim_sep(vocab));
            try extra_tokens.appendSlice(a, k_fim_file);
        } else {
            // chunk separator in binary form to avoid confusing the AI
            const k_chunk_prefix_str = [_]u8{ 0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70, 0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a };
            const k_chunk_prefix_tokens = try infill_statics.get(&infill_statics.k_chunk_prefix_tokens, vocab, &k_chunk_prefix_str);

            try extra_tokens.appendSlice(a, k_chunk_prefix_tokens);
        }

        const chunk_tokens = try common.tokenize(a, vocab, text, false, false);
        defer a.free(chunk_tokens);
        try extra_tokens.appendSlice(a, chunk_tokens);
    }

    if (c.llama_vocab_fim_sep(vocab) != token_null) {
        // TODO: current filename
        const k_fim_file = try infill_statics.get(&infill_statics.k_fim_file, vocab, "filename\n");

        try extra_tokens.append(a, c.llama_vocab_fim_sep(vocab));
        try extra_tokens.appendSlice(a, k_fim_file);
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const n_prefix_take: c_int = @min(@as(c_int, @truncate(@as(i64, @intCast(tokens_prefix.items.len)))), 3 * @divTrunc(n_batch, 4));
    // `(n_batch/4) - (2 + tokens_prompt.size())` is unsigned, then `int`
    const wrapped: u64 = @as(u64, @bitCast(@as(i64, @divTrunc(n_batch, 4)))) -% (2 + @as(u64, tokens_prompt.len));
    const n_suffix_take: c_int = @min(@as(c_int, @truncate(@as(i64, @intCast(tokens_suffix.items.len)))), @max(0, @as(c_int, @truncate(@as(i64, @bitCast(wrapped))))));

    common.srv(.dbg, "format_prompt_infill", "n_prefix_take = %d, n_suffix_take = %d, total = %d\n", .{ n_prefix_take, n_suffix_take, (n_prefix_take + n_suffix_take) });

    // fill the rest of the context with extra chunks
    const n_extra_take: c_int = @min(@max(0, n_ctx - (n_batch) - 2 * n_predict), @as(c_int, @truncate(@as(i64, @intCast(extra_tokens.items.len)))));

    const drop: usize = tokens_prefix.items.len - @as(usize, @intCast(n_prefix_take));
    tokens_prefix.replaceRangeAssumeCapacity(0, drop, &.{});
    try tokens_suffix.resize(a, @intCast(n_suffix_take));

    try tokens_prefix.insert(a, 0, c.llama_vocab_fim_pre(vocab));
    try tokens_prefix.appendSlice(a, tokens_prompt);
    try tokens_suffix.insert(a, 0, c.llama_vocab_fim_suf(vocab));

    const inp_src = if (spm_infill) tokens_suffix.items else tokens_prefix.items;
    const end_src = if (spm_infill) tokens_prefix.items else tokens_suffix.items;
    var embd_inp: std.ArrayList(Token) = .empty;
    errdefer embd_inp.deinit(a);
    try embd_inp.appendSlice(a, inp_src);

    if (c.llama_vocab_get_add_bos(vocab)) {
        try embd_inp.insert(a, 0, c.llama_vocab_bos(vocab));
    }

    common.srv(.dbg, "format_prompt_infill", "extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", .{ n_ctx, n_extra_take, @as(c_int, @intCast(extra_tokens.items.len)) });

    // put the extra context before the FIM prefix
    try embd_inp.insertSlice(a, 0, extra_tokens.items[extra_tokens.items.len - @as(usize, @intCast(n_extra_take)) ..]);

    try embd_inp.appendSlice(a, end_src);
    try embd_inp.append(a, c.llama_vocab_fim_mid(vocab));

    return embd_inp.toOwnedSlice(a);
}

/// Ports `string_replace_all` (common/common.cpp:510 @d81235049).
///
/// Return: owned by `a`.
pub fn replaceAll(a: Allocator, s: []const u8, search: []const u8, replace: []const u8) Error![]u8 {
    if (search.len == 0) return a.dupe(u8, s);
    var o: std.ArrayList(u8) = .empty;
    errdefer o.deinit(a);
    var last: usize = 0;
    while (std.mem.indexOfPos(u8, s, last, search)) |pos| {
        try o.appendSlice(a, s[last..pos]);
        try o.appendSlice(a, replace);
        last = pos + search.len;
    }
    try o.appendSlice(a, s[last..]);
    return o.toOwnedSlice(a);
}

/// Ports `format_prompt_rerank` (server-common.cpp:1880 @d81235049): the
/// model's `rerank` template with `{query}` and `{document}` filled, or
/// `[BOS]query[EOS][SEP]doc[EOS]`.
pub fn formatPromptRerank(
    gpa: Allocator,
    model: ?*const c.llama_model,
    vocab: ?*const c.llama_vocab,
    mctx: ?*c.mtmd_context,
    query: []const u8,
    doc: []const u8,
    init_opt: c.mtmd_helper_init_opt,
) Error!ServerTokens {
    var result = ServerTokens.init(gpa);
    errdefer result.deinit();

    const rerank_prompt: ?[*:0]const u8 = c.llama_model_chat_template(model, "rerank");

    if (rerank_prompt) |rp| {
        const p1 = try replaceAll(gpa, std.mem.span(rp), "{query}", query);
        defer gpa.free(p1);
        const prompt = try replaceAll(gpa, p1, "{document}", doc);
        var pv: Value = .{ .string = prompt };
        defer pv.deinit(gpa);
        var tokens = try tokenizeInputSubprompt(gpa, vocab, mctx, &pv, false, true, init_opt);
        defer tokens.deinit();
        try result.pushBackTokens(&tokens);
    } else {
        // Get EOS token - use SEP token as fallback if EOS is not available
        var qv: Value = .{ .string = try gpa.dupe(u8, query) };
        defer qv.deinit(gpa);
        var dv: Value = .{ .string = try gpa.dupe(u8, doc) };
        defer dv.deinit(gpa);
        var query_tokens = try tokenizeInputSubprompt(gpa, vocab, mctx, &qv, false, false, init_opt);
        defer query_tokens.deinit();
        var doc_tokens = try tokenizeInputSubprompt(gpa, vocab, mctx, &dv, false, false, init_opt);
        defer doc_tokens.deinit();
        var eos_token = c.llama_vocab_eos(vocab);
        if (eos_token == token_null) {
            eos_token = c.llama_vocab_sep(vocab);
        }

        if (c.llama_vocab_get_add_bos(vocab)) {
            try result.pushBackToken(c.llama_vocab_bos(vocab));
        }
        try result.pushBackTokens(&query_tokens);
        if (c.llama_vocab_get_add_eos(vocab)) {
            try result.pushBackToken(eos_token);
        }
        if (c.llama_vocab_get_add_sep(vocab)) {
            try result.pushBackToken(c.llama_vocab_sep(vocab));
        }
        try result.pushBackTokens(&doc_tokens);
        if (c.llama_vocab_get_add_eos(vocab)) {
            try result.pushBackToken(eos_token);
        }
    }

    return result;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "base64_decode stops at padding and drops a partial quantum" {
    const a = std.testing.allocator;
    const d = try base64Decode(a, "aGVsbG8=");
    defer a.free(d);
    try std.testing.expectEqualStrings("hello", d);
    const e = try base64Decode(a, "aGVsbA!xyz");
    defer a.free(e);
    try std.testing.expectEqualStrings("hell", e);
}

test "a plain token list round-trips through the state format" {
    const a = std.testing.allocator;
    var t = try ServerTokens.fromTokens(a, &.{ 5, 6, 7 }, false);
    defer t.deinit();
    const bytes = try t.serialize(a);
    defer a.free(bytes);
    try std.testing.expectEqual(@as(usize, 0), bytes.len % 4);
    const toks = std.mem.bytesAsSlice(Token, bytes);
    const copy = try a.dupe(Token, @alignCast(toks));
    defer a.free(copy);
    var back = try ServerTokens.deserialize(a, copy, false);
    defer back.deinit();
    try std.testing.expectEqualSlices(Token, t.tokens.items, back.tokens.items);
}
