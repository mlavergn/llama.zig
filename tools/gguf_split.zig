//! `llama-gguf-split`: split a GGUF into shards, or merge shards back.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/gguf-split/gguf-split.cpp` at v0.3.0
//! (`c1d0e7a00`). Each declaration names the C++ it replaces and the line it
//! began at.
//!
//! # Upstream's output, byte for byte
//!
//! Messages go through C stdio with the C's own formats, `stdout` and
//! `stderr` as the C chose each, and the files are written in the C's order:
//! a split's metadata then each tensor padded to `GGUF_DEFAULT_ALIGNMENT`; a
//! merge's metadata as a zeroed placeholder, the tensors, then the metadata
//! over the placeholder. `scripts/tools-diff` holds both to upstream's
//! binary.
//!
//! # Streams
//!
//! `std::ifstream` and `std::ofstream` are C `FILE *`s here. The C++ sets
//! `failbit` exceptions on the output stream and never catches them, so a
//! failed write terminates the process; here it aborts.

const std = @import("std");
const builtin = @import("builtin");
const common = @import("tools_common");
const c = common.c;
const io = common.io;

comptime {
    // libllama, compiled in (see `build/tools.zig`).
    _ = @import("llama");
}

/// `PATH_MAX`, or `MAX_PATH` on Windows (gguf-split.cpp:24 @c1d0e7a00).
const path_max = if (builtin.os.tag == .windows) 260 else if (builtin.os.tag.isDarwin()) 1024 else 4096;

/// Mirrors `split_operation` (gguf-split.cpp:29 @c1d0e7a00).
const Operation = enum { none, split, merge };

/// Mirrors `split_mode` (gguf-split.cpp:35 @c1d0e7a00).
const Mode = enum { none, tensor, size };

/// Mirrors `split_params` (gguf-split.cpp:41 @c1d0e7a00).
const Params = struct {
    operation: Operation = .none,
    mode: Mode = .none,
    n_bytes_split: usize = 0,
    n_split_tensors: c_int = 128,
    input: [:0]const u8 = "",
    output: [:0]const u8 = "",
    no_tensor_first_split: bool = false,
    dry_run: bool = false,
    delete_splits: bool = false,
};

/// Ports `split_print_usage` (gguf-split.cpp:53 @c1d0e7a00).
fn printUsage(executable: [*:0]const u8) void {
    const default_params: Params = .{};
    io.printf("\n", .{});
    io.printf("usage: %s [options] GGUF_IN GGUF_OUT\n", .{executable});
    io.printf("\n", .{});
    io.printf("Apply a GGUF operation on IN to OUT.", .{});
    io.printf("\n", .{});
    io.printf("options:\n", .{});
    io.printf("  -h, --help              show this help message and exit\n", .{});
    io.printf("  --version               show version and build info\n", .{});
    io.printf("  --split                 split GGUF to multiple GGUF (enabled by default)\n", .{});
    io.printf("  --merge                 merge multiple GGUF to a single GGUF\n", .{});
    io.printf("  --split-max-tensors     max tensors in each split (default: %d)\n", .{default_params.n_split_tensors});
    io.printf("  --split-max-size N(M|G) max size per split\n", .{});
    io.printf("  --no-tensor-first-split do not add tensors to the first split (disabled by default)\n", .{});
    io.printf("  --dry-run               only print out a split plan and exit, without writing any new files\n", .{});
    io.printf("  --delete-splits         delete the split files during merge to free up disk space WARNING: this option is unsafe and will leave you in an unrecoverable state if something fails during the merge\n", .{});
    io.printf("\n", .{});
}

/// What the C++ throws as `std::invalid_argument`: the message is printed,
/// then the usage, and the process exits 1, as `split_params_parse`
/// (gguf-split.cpp:183 @c1d0e7a00) does.
const ParseError = struct {
    buf: [512]u8 = undefined,
    len: usize = 0,

    fn set(self: *ParseError, comptime fmt: []const u8, args: anytype) error{Invalid} {
        const s = std.fmt.bufPrint(&self.buf, fmt, args) catch self.buf[0..];
        self.len = s.len;
        return error.Invalid;
    }
};

/// `sscanf(str, "%d", &n)`: an optional sign and decimal digits after
/// optional white space, or no conversion.
fn scanInt(s: []const u8) ?c_int {
    var i: usize = 0;
    while (i < s.len and std.ascii.isWhitespace(s[i])) i += 1;
    const start = i;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
    const digits = i;
    while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    if (i == digits) return null;
    const v = std.fmt.parseInt(i64, s[start..i], 10) catch return null;
    return @truncate(v);
}

/// Ports `split_str_to_n_bytes` (gguf-split.cpp:74 @c1d0e7a00).
///
/// Where `sscanf` converts nothing the C reads `n` uninitialised; here that
/// is reported as a non-positive size, the C's next check.
fn strToNBytes(err: *ParseError, str: []const u8) error{Invalid}!usize {
    if (str.len == 0) return err.set("error: supported units are M (megabytes) or G (gigabytes), but got: \x00", .{});
    var n_bytes: usize = 0;
    var n: c_int = 0;
    const last = str[str.len - 1];
    if (last == 'M') {
        n = scanInt(str) orelse 0;
        n_bytes = @as(usize, @bitCast(@as(isize, n))) *% 1000 *% 1000; // megabytes
    } else if (last == 'G') {
        n = scanInt(str) orelse 0;
        n_bytes = @as(usize, @bitCast(@as(isize, n))) *% 1000 *% 1000 *% 1000; // gigabytes
    } else {
        return err.set("error: supported units are M (megabytes) or G (gigabytes), but got: {c}", .{last});
    }
    if (n <= 0) return err.set("error: size must be a positive value", .{});
    return n_bytes;
}

/// Ports `split_params_parse_ex` (gguf-split.cpp:92 @c1d0e7a00).
fn parseEx(argv: []const [:0]const u8, params: *Params, err: *ParseError, arena: std.mem.Allocator) error{ Invalid, OutOfMemory }!void {
    var arg: []u8 = &.{};
    var invalid_param = false;

    var arg_idx: usize = 1;
    while (arg_idx < argv.len and std.mem.startsWith(u8, argv[arg_idx], "--")) : (arg_idx += 1) {
        arg = try arena.dupe(u8, argv[arg_idx]);
        std.mem.replaceScalar(u8, arg, '_', '-');

        var arg_found = false;
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            printUsage(argv[0].ptr);
            common.cExit(0);
        } else if (std.mem.eql(u8, arg, "--version")) {
            common.build_info.print(c.llama_version());
            common.cExit(0);
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            arg_found = true;
            params.dry_run = true;
        } else if (std.mem.eql(u8, arg, "--no-tensor-first-split")) {
            arg_found = true;
            params.no_tensor_first_split = true;
        } else if (std.mem.eql(u8, arg, "--merge")) {
            arg_found = true;
            if (params.operation != .none and params.operation != .merge) {
                return err.set("error: either --split or --merge can be specified, but not both", .{});
            }
            params.operation = .merge;
        } else if (std.mem.eql(u8, arg, "--split")) {
            arg_found = true;
            if (params.operation != .none and params.operation != .split) {
                return err.set("error: either --split or --merge can be specified, but not both", .{});
            }
            params.operation = .split;
        } else if (std.mem.eql(u8, arg, "--split-max-tensors")) {
            arg_idx += 1;
            if (arg_idx >= argv.len) {
                invalid_param = true;
                break;
            }
            arg_found = true;
            if (params.mode != .none and params.mode != .tensor) {
                return err.set("error: either --split-max-tensors or --split-max-size can be specified, but not both", .{});
            }
            params.mode = .tensor;
            params.n_split_tensors = c.atoi(argv[arg_idx].ptr);
        } else if (std.mem.eql(u8, arg, "--split-max-size")) {
            arg_idx += 1;
            if (arg_idx >= argv.len) {
                invalid_param = true;
                break;
            }
            arg_found = true;
            if (params.mode != .none and params.mode != .size) {
                return err.set("error: either --split-max-tensors or --split-max-size can be specified, but not both", .{});
            }
            params.mode = .size;
            params.n_bytes_split = try strToNBytes(err, argv[arg_idx]);
        } else if (std.mem.eql(u8, arg, "--delete-splits")) {
            arg_found = true;
            params.delete_splits = true;
        }

        if (!arg_found) return err.set("error: unknown argument: {s}", .{arg});
    }

    // the operation is split if not specified
    if (params.operation == .none) params.operation = .split;
    // the split mode is by tensor if not specified
    if (params.mode == .none) params.mode = .tensor;

    if (invalid_param) return err.set("error: invalid parameter for argument: {s}", .{arg});

    if (argv.len - arg_idx != 2) return err.set("error: bad arguments", .{});

    params.input = argv[arg_idx];
    params.output = argv[arg_idx + 1];
}

/// Ports `split_params_parse` (gguf-split.cpp:183 @c1d0e7a00).
fn parse(argv: []const [:0]const u8, params: *Params, arena: std.mem.Allocator) !void {
    var err: ParseError = .{};
    parseEx(argv, params, &err, arena) catch |e| switch (e) {
        error.Invalid => {
            io.eprintf("%.*s\n", .{ @as(c_int, @intCast(err.len)), &err.buf });
            printUsage(argv[0].ptr);
            common.cExit(1);
        },
        else => |other| return other,
    };
}

// -----------------------------------------------------------------------------
// C stdio, as the C++ streams

extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern fn fclose(f: *anyopaque) c_int;
extern fn fread(ptr: [*]u8, size: usize, n: usize, f: *anyopaque) usize;
extern fn fwrite(ptr: [*]const u8, size: usize, n: usize, f: *anyopaque) usize;
extern fn fseeko(f: *anyopaque, off: i64, whence: c_int) c_int;
extern fn _fseeki64(f: *anyopaque, off: i64, whence: c_int) c_int;
extern fn remove(path: [*:0]const u8) c_int;

fn seek(f: *anyopaque, off: u64) void {
    const r = if (comptime builtin.os.tag == .windows) _fseeki64(f, @intCast(off), 0) else fseeko(f, @intCast(off), 0);
    _ = r;
}

/// `fout.write(..)` with `failbit` exceptions: a failed write terminates.
fn writeAll(f: *anyopaque, bytes: []const u8) void {
    if (bytes.len == 0) return;
    if (fwrite(bytes.ptr, 1, bytes.len, f) != bytes.len) {
        io.eprintf("terminate called after throwing an instance of 'std::ios_base::failure'\n", .{});
        std.process.abort();
    }
}

/// `f_in.read(..)`: a short read leaves the rest of the buffer as it was.
fn readInto(f: *anyopaque, buf: []u8) void {
    _ = fread(buf.ptr, 1, buf.len, f);
}

/// Ports `zeros` (gguf-split.cpp:196 @c1d0e7a00).
fn zeros(f: *anyopaque, n: usize) void {
    const zero = [1]u8{0};
    for (0..n) |_| writeAll(f, &zero);
}

/// A path buffer as the `char *` the C passes to `printf`.
fn cstr(buf: *const [path_max:0]u8) [*:0]const u8 {
    return buf;
}

/// `GGML_PAD(x, GGUF_DEFAULT_ALIGNMENT)`.
fn pad(x: usize) usize {
    const a: usize = c.GGUF_DEFAULT_ALIGNMENT;
    return (x + a - 1) & ~(a - 1);
}

// -----------------------------------------------------------------------------
// Split

/// Ports `split_strategy` (gguf-split.cpp:203 @c1d0e7a00).
const Strategy = struct {
    params: Params,
    f_input: *anyopaque,
    ctx_gguf: *c.gguf_context,
    ctx_meta: *c.ggml_context,
    n_tensors: c_int,
    // one ctx_out per one output file
    ctx_outs: std.ArrayList(*c.gguf_context) = .empty,
    // temporary buffer for reading in tensor data
    read_buf: std.ArrayList(u8) = .empty,
    allocator: std.mem.Allocator,

    // because we need to know list of tensors for each file in advance, we
    // will build all the ctx_out for all output splits
    fn init(allocator: std.mem.Allocator, params: Params, f_input: *anyopaque, ctx_gguf: *c.gguf_context, ctx_meta: *c.ggml_context) !Strategy {
        var self: Strategy = .{
            .params = params,
            .f_input = f_input,
            .ctx_gguf = ctx_gguf,
            .ctx_meta = ctx_meta,
            .n_tensors = @intCast(c.gguf_get_n_tensors(ctx_gguf)),
            .allocator = allocator,
        };

        var i_split: c_int = -1;
        var ctx_out: ?*c.gguf_context = null;

        // initialize ctx_out for the first split
        try self.newCtxOut(&i_split, &ctx_out, false);

        // skip first split if no_tensor_first_split is set
        if (params.no_tensor_first_split) try self.newCtxOut(&i_split, &ctx_out, true);

        // process tensors one by one
        var curr_tensors_size: usize = 0; // current size by counting only tensors size (without metadata)
        var i: c_int = 0;
        while (i < self.n_tensors) : (i += 1) {
            const t = c.ggml_get_tensor(ctx_meta, c.gguf_get_tensor_name(ctx_gguf, i));
            // calculate the "imaginary" size = the current size + next tensor size
            const n_bytes = pad(c.ggml_nbytes(t));
            const next_tensors_size = curr_tensors_size + n_bytes;
            if (self.shouldSplit(i, next_tensors_size)) {
                try self.newCtxOut(&i_split, &ctx_out, false);
                curr_tensors_size = n_bytes;
            } else {
                curr_tensors_size = next_tensors_size;
            }
            c.gguf_add_tensor(ctx_out.?, t);
        }

        // push the last ctx_out
        try self.ctx_outs.append(allocator, ctx_out.?);

        // set the correct n_split for all ctx_out
        for (self.ctx_outs.items) |ctx| c.gguf_set_val_u16(ctx, common.kv_split_count, @truncate(self.ctx_outs.items.len));
        return self;
    }

    /// The C++ constructor's `new_ctx_out` lambda (gguf-split.cpp:229 @c1d0e7a00).
    fn newCtxOut(self: *Strategy, i_split: *c_int, ctx_out: *?*c.gguf_context, allow_no_tensors: bool) !void {
        i_split.* += 1;
        if (ctx_out.*) |prev| {
            if (c.gguf_get_n_tensors(prev) == 0 and !allow_no_tensors) {
                io.eprintf("error: one of splits have 0 tensors. Maybe size or tensors limit is too small\n", .{});
                common.cExit(1);
            }
            try self.ctx_outs.append(self.allocator, prev);
        }
        const out = c.gguf_init_empty().?;
        ctx_out.* = out;
        // Save all metadata in first split only
        if (i_split.* == 0) c.gguf_set_kv(out, self.ctx_gguf);
        c.gguf_set_val_u16(out, common.kv_split_no, @truncate(@as(c_uint, @bitCast(i_split.*))));
        c.gguf_set_val_u16(out, common.kv_split_count, 0); // placeholder
        c.gguf_set_val_i32(out, common.kv_split_tensors_count, self.n_tensors);
    }

    fn deinit(self: *Strategy) void {
        for (self.ctx_outs.items) |ctx_out| c.gguf_free(ctx_out);
        self.ctx_outs.deinit(self.allocator);
        self.read_buf.deinit(self.allocator);
    }

    /// Ports `should_split` (gguf-split.cpp:287 @c1d0e7a00).
    fn shouldSplit(self: *const Strategy, i_tensor: c_int, next_size: usize) bool {
        return switch (self.params.mode) {
            // split by max size per file
            .size => next_size > self.params.n_bytes_split,
            // split by number of tensors per file
            .tensor => i_tensor > 0 and i_tensor < self.n_tensors and @rem(i_tensor, self.params.n_split_tensors) == 0,
            // should never happen
            .none => c.ggml_abort("gguf-split.cpp", 297, "invalid mode"),
        };
    }

    /// Ports `print_info` (gguf-split.cpp:299 @c1d0e7a00).
    fn printInfo(self: *const Strategy) void {
        io.printf("n_split: %zu\n", .{self.ctx_outs.items.len});
        var i_split: c_int = 0;
        for (self.ctx_outs.items) |ctx_out| {
            // re-calculate the real gguf size for each split (= metadata size + total size of all tensors)
            var total_size: usize = c.gguf_get_meta_size(ctx_out);
            var i: i64 = 0;
            while (i < c.gguf_get_n_tensors(ctx_out)) : (i += 1) {
                const t = c.ggml_get_tensor(self.ctx_meta, c.gguf_get_tensor_name(ctx_out, i));
                total_size += c.ggml_nbytes(t);
            }
            total_size = total_size / 1000 / 1000; // convert to megabytes
            io.printf("split %05d: n_tensors = %lld, total_size = %zuM\n", .{ i_split + 1, @as(c_longlong, c.gguf_get_n_tensors(ctx_out)), total_size });
            i_split += 1;
        }
    }

    /// Ports `write` (gguf-split.cpp:315 @c1d0e7a00).
    fn write(self: *Strategy) !void {
        var i_split: c_int = 0;
        const n_split: c_int = @intCast(self.ctx_outs.items.len);
        for (self.ctx_outs.items) |ctx_out| {
            // construct file path
            var split_path: [path_max:0]u8 = @splat(0);
            _ = c.llama_split_path(&split_path, split_path.len, self.params.output.ptr, i_split, n_split);

            // open the output file
            io.printf("Writing file %s ... ", .{cstr(&split_path)});
            io.flushStdout();
            const fout = fopen(&split_path, "wb") orelse {
                io.eprintf("terminate called after throwing an instance of 'std::ios_base::failure'\n", .{});
                std.process.abort();
            };

            // write metadata
            const data = try self.allocator.alloc(u8, c.gguf_get_meta_size(ctx_out));
            defer self.allocator.free(data);
            c.gguf_get_meta_data(ctx_out, data.ptr);
            writeAll(fout, data);

            // write tensors
            var i: i64 = 0;
            while (i < c.gguf_get_n_tensors(ctx_out)) : (i += 1) {
                // read tensor meta and prepare buffer
                const t_name = c.gguf_get_tensor_name(ctx_out, i);
                const t = c.ggml_get_tensor(self.ctx_meta, t_name);
                const n_bytes = c.ggml_nbytes(t);
                try self.read_buf.resize(self.allocator, n_bytes);

                // calculate offset
                const i_tensor_in = c.gguf_find_tensor(self.ctx_gguf, t_name); // idx of tensor in the input file
                const offset = c.gguf_get_data_offset(self.ctx_gguf) + c.gguf_get_tensor_offset(self.ctx_gguf, i_tensor_in);

                // copy tensor from input to output file
                try self.copyFileToFile(fout, offset, n_bytes);
                zeros(fout, pad(n_bytes) - n_bytes);
            }

            io.printf("done\n", .{});
            // close the file
            _ = fclose(fout);
            i_split += 1;
        }
    }

    /// Ports `copy_file_to_file` (gguf-split.cpp:358 @c1d0e7a00).
    fn copyFileToFile(self: *Strategy, f_out: *anyopaque, in_offset: usize, len: usize) !void {
        if (self.read_buf.items.len < len) try self.read_buf.resize(self.allocator, len);
        seek(self.f_input, in_offset);
        readInto(self.f_input, self.read_buf.items[0..len]);
        writeAll(f_out, self.read_buf.items[0..len]);
    }
};

/// Ports `gguf_split` (gguf-split.cpp:369 @c1d0e7a00).
fn ggufSplit(allocator: std.mem.Allocator, split_params: Params) !void {
    var ctx_meta: ?*c.ggml_context = null;

    const params: c.gguf_init_params = .{ .no_alloc = true, .ctx = &ctx_meta };

    const f_input = fopen(split_params.input.ptr, "rb") orelse {
        io.eprintf("%s:  failed to open input GGUF from %s\n", .{ "gguf_split", split_params.input.ptr });
        common.cExit(1);
    };

    const ctx_gguf = c.gguf_init_from_file(split_params.input.ptr, params) orelse {
        io.eprintf("%s:  failed to load input GGUF from %s\n", .{ "gguf_split", split_params.input.ptr });
        common.cExit(1);
    };

    // prepare the strategy
    var strategy = try Strategy.init(allocator, split_params, f_input, ctx_gguf, ctx_meta.?);
    defer strategy.deinit();
    const n_split: c_int = @intCast(strategy.ctx_outs.items.len);
    strategy.printInfo();

    if (!split_params.dry_run) {
        // write all output splits
        try strategy.write();
    }

    // done, clean up
    c.gguf_free(ctx_gguf);
    _ = fclose(f_input);

    io.eprintf("%s: %d gguf split written with a total of %d tensors.\n", .{ "gguf_split", n_split, strategy.n_tensors });
}

/// Ports `gguf_merge` (gguf-split.cpp:407 @c1d0e7a00).
fn ggufMerge(allocator: std.mem.Allocator, split_params: Params) !void {
    const func = "gguf_merge";
    io.eprintf("%s: %s -> %s\n", .{ func, split_params.input.ptr, split_params.output.ptr });
    var n_split: c_int = 1;
    var total_tensors: c_int = 0;

    // avoid overwriting existing output file
    if (fopen(split_params.output.ptr, "rb")) |existing| {
        _ = fclose(existing);
        io.eprintf("%s: output file %s already exists\n", .{ func, split_params.output.ptr });
        common.cExit(1);
    }

    const ctx_out = c.gguf_init_empty().?;

    var read_data: std.ArrayList(u8) = .empty;
    defer read_data.deinit(allocator);
    var ctx_metas: std.ArrayList(*c.ggml_context) = .empty;
    defer ctx_metas.deinit(allocator);
    var ctx_ggufs: std.ArrayList(*c.gguf_context) = .empty;
    defer ctx_ggufs.deinit(allocator);

    var split_path: [path_max:0]u8 = @splat(0);
    const in_len = @min(split_params.input.len, split_path.len - 1);
    @memcpy(split_path[0..in_len], split_params.input[0..in_len]);
    var split_prefix: [path_max:0]u8 = @splat(0);

    // First pass to find KV and tensors metadata
    var i_split: c_int = 0;
    while (i_split < n_split) : (i_split += 1) {
        var ctx_meta: ?*c.ggml_context = null;

        const params: c.gguf_init_params = .{ .no_alloc = true, .ctx = &ctx_meta };

        if (i_split > 0) _ = c.llama_split_path(&split_path, split_path.len, &split_prefix, i_split, n_split);
        io.eprintf("%s: reading metadata %s ...", .{ func, cstr(&split_path) });

        const ctx_gguf = c.gguf_init_from_file(&split_path, params) orelse {
            io.eprintf("\n%s:  failed to load input GGUF from %s\n", .{ func, split_params.input.ptr });
            common.cExit(1);
        };
        try ctx_ggufs.append(allocator, ctx_gguf);
        try ctx_metas.append(allocator, ctx_meta.?);

        if (i_split == 0) {
            const key_n_split = c.gguf_find_key(ctx_gguf, common.kv_split_count);
            if (key_n_split < 0) {
                io.eprintf("\n%s: input file does not contain %s metadata\n", .{ func, common.kv_split_count.ptr });
                c.gguf_free(ctx_gguf);
                c.ggml_free(ctx_meta);
                c.gguf_free(ctx_out);
                common.cExit(1);
            }

            n_split = c.gguf_get_val_u16(ctx_gguf, key_n_split);
            if (n_split < 1) {
                io.eprintf("\n%s: input file does not contain a valid split count %d\n", .{ func, n_split });
                c.gguf_free(ctx_gguf);
                c.ggml_free(ctx_meta);
                c.gguf_free(ctx_out);
                common.cExit(1);
            }

            // Verify the file naming and extract split_prefix
            if (c.llama_split_prefix(&split_prefix, split_prefix.len, &split_path, i_split, n_split) == 0) {
                io.eprintf("\n%s: unexpected input file name: %s i_split=%d n_split=%d\n", .{ func, cstr(&split_path), i_split, n_split });
                c.gguf_free(ctx_gguf);
                c.ggml_free(ctx_meta);
                c.gguf_free(ctx_out);
                common.cExit(1);
            }

            // Do not trigger merge if we try to merge again the output
            c.gguf_set_val_u16(ctx_gguf, common.kv_split_count, 0);

            // Set metadata from the first split
            c.gguf_set_kv(ctx_out, ctx_gguf);
        }

        const n_tensors = c.gguf_get_n_tensors(ctx_gguf);
        var i_tensor: i64 = 0;
        while (i_tensor < n_tensors) : (i_tensor += 1) {
            const t_name = c.gguf_get_tensor_name(ctx_gguf, i_tensor);
            const t = c.ggml_get_tensor(ctx_meta, t_name);
            c.gguf_add_tensor(ctx_out, t);
        }
        total_tensors += @intCast(n_tensors);

        io.eprintf("\x1b[3Ddone\n", .{});
    }

    var fout: ?*anyopaque = null;
    if (!split_params.dry_run) {
        fout = fopen(split_params.output.ptr, "wb") orelse {
            io.eprintf("terminate called after throwing an instance of 'std::ios_base::failure'\n", .{});
            std.process.abort();
        };
        // placeholder for the meta data
        const meta_size = c.gguf_get_meta_size(ctx_out);
        zeros(fout.?, meta_size);
    }

    // Write tensors data
    var merge_error = false;
    i_split = 0;
    while (i_split < n_split) : (i_split += 1) {
        _ = c.llama_split_path(&split_path, split_path.len, &split_prefix, i_split, n_split);
        const f_input = fopen(&split_path, "rb") orelse {
            io.eprintf("%s:  failed to open input GGUF from %s\n", .{ func, cstr(&split_path) });
            for (ctx_ggufs.items, ctx_metas.items) |g, m| {
                c.gguf_free(g);
                c.ggml_free(m);
            }
            c.gguf_free(ctx_out);
            if (!split_params.dry_run) _ = fclose(fout.?);
            common.cExit(1);
        };
        io.eprintf("%s: writing tensors %s ...", .{ func, cstr(&split_path) });

        const ctx_gguf = ctx_ggufs.items[@intCast(i_split)];
        const ctx_meta = ctx_metas.items[@intCast(i_split)];

        const n_tensors = c.gguf_get_n_tensors(ctx_gguf);
        var i_tensor: i64 = 0;
        while (i_tensor < n_tensors) : (i_tensor += 1) {
            const t_name = c.gguf_get_tensor_name(ctx_gguf, i_tensor);
            const t = c.ggml_get_tensor(ctx_meta, t_name);

            const n_bytes = c.ggml_nbytes(t);

            if (read_data.items.len < n_bytes) try read_data.resize(allocator, n_bytes);

            const offset = c.gguf_get_data_offset(ctx_gguf) + c.gguf_get_tensor_offset(ctx_gguf, i_tensor);
            seek(f_input, offset);
            readInto(f_input, read_data.items[0..n_bytes]);
            if (!split_params.dry_run) {
                // write tensor data + padding
                writeAll(fout.?, read_data.items[0..n_bytes]);
                zeros(fout.?, pad(n_bytes) - n_bytes);
            }
        }

        c.gguf_free(ctx_gguf);
        c.ggml_free(ctx_meta);
        _ = fclose(f_input);
        io.eprintf("\x1b[3Ddone\n", .{});

        if (!split_params.dry_run and split_params.delete_splits) {
            const delete_result = remove(&split_path);
            if (delete_result != 0) {
                merge_error = true;
                io.eprintf("error: failed to delete %s\n", .{cstr(&split_path)});
            } else {
                io.eprintf("%s: deleted file %s\n", .{ func, cstr(&split_path) });
            }
        }
    }

    if (!split_params.dry_run) {
        // go back to beginning of file and write the updated metadata
        seek(fout.?, 0);
        const data = try allocator.alloc(u8, c.gguf_get_meta_size(ctx_out));
        defer allocator.free(data);
        c.gguf_get_meta_data(ctx_out, data.ptr);
        writeAll(fout.?, data);
        _ = fclose(fout.?);
    }
    c.gguf_free(ctx_out);

    io.eprintf("%s: %s merged from %d split with %d tensors.\n", .{ func, split_params.output.ptr, n_split, total_tensors });

    if (merge_error) common.cExit(1);
}

/// Ports `main` (gguf-split.cpp:593 @c1d0e7a00).
pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);

    var params: Params = .{};
    try parse(argv, &params, arena);

    switch (params.operation) {
        .split => try ggufSplit(init.gpa, params),
        .merge => try ggufMerge(init.gpa, params),
        else => {
            printUsage(argv[0].ptr);
            common.cExit(1);
        },
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "split sizes parse as the C's sscanf reads them" {
    var err: ParseError = .{};
    try std.testing.expectEqual(@as(usize, 128_000_000), try strToNBytes(&err, "128M"));
    try std.testing.expectEqual(@as(usize, 4_000_000_000), try strToNBytes(&err, "4G"));
    try std.testing.expectError(error.Invalid, strToNBytes(&err, "4K"));
    try std.testing.expectError(error.Invalid, strToNBytes(&err, "0M"));
}
