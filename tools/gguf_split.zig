//! `llama-gguf-split`: split a GGUF into shards, or merge shards back.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/gguf-split/gguf-split.cpp` at v1.0.0
//! (`b1b770d90`). Each declaration names the C++ it replaces and the line it
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
//! `std::ifstream` and `std::ofstream` are `tools_common.file`'s. The C++
//! sets `failbit` exceptions on the output stream and never catches them,
//! so a failed write terminates the process; here it aborts.

const std = @import("std");
const llama = @import("llama");
const stdlib = @import("stdlib");
const builtin = @import("builtin");
const tools_common = @import("tools_common");

/// `PATH_MAX`, or `MAX_PATH` on Windows (gguf-split.cpp:24 @b1b770d90).
const path_max = if (builtin.os.tag == .windows) 260 else if (builtin.os.tag.isDarwin()) 1024 else 4096;

/// Mirrors `split_operation` (gguf-split.cpp:29 @b1b770d90).
const Operation = enum { none, split, merge };

/// Mirrors `split_mode` (gguf-split.cpp:35 @b1b770d90).
const Mode = enum { none, tensor, size };

/// Mirrors `split_params` (gguf-split.cpp:41 @b1b770d90).
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

/// Ports `split_print_usage` (gguf-split.cpp:53 @b1b770d90).
fn printUsage(executable: [*:0]const u8) void {
    const default_params: Params = .{};
    tools_common.io.printf("\n", .{});
    tools_common.io.printf("usage: %s [options] GGUF_IN GGUF_OUT\n", .{executable});
    tools_common.io.printf("\n", .{});
    tools_common.io.printf("Apply a GGUF operation on IN to OUT.", .{});
    tools_common.io.printf("\n", .{});
    tools_common.io.printf("options:\n", .{});
    tools_common.io.printf("  -h, --help              show this help message and exit\n", .{});
    tools_common.io.printf("  --version               show version and build info\n", .{});
    tools_common.io.printf("  --split                 split GGUF to multiple GGUF (enabled by default)\n", .{});
    tools_common.io.printf("  --merge                 merge multiple GGUF to a single GGUF\n", .{});
    tools_common.io.printf("  --split-max-tensors     max tensors in each split (default: %d)\n", .{default_params.n_split_tensors});
    tools_common.io.printf("  --split-max-size N(M|G) max size per split\n", .{});
    tools_common.io.printf("  --no-tensor-first-split do not add tensors to the first split (disabled by default)\n", .{});
    tools_common.io.printf("  --dry-run               only print out a split plan and exit, without writing any new files\n", .{});
    tools_common.io.printf("  --delete-splits         delete the split files during merge to free up disk space WARNING: this option is unsafe and will leave you in an unrecoverable state if something fails during the merge\n", .{});
    tools_common.io.printf("\n", .{});
}

/// What the C++ throws as `std::invalid_argument`: the message is printed,
/// then the usage, and the process exits 1, as `split_params_parse`
/// (gguf-split.cpp:183 @b1b770d90) does.
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

/// Ports `split_str_to_n_bytes` (gguf-split.cpp:74 @b1b770d90).
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

/// Ports `split_params_parse_ex` (gguf-split.cpp:92 @b1b770d90).
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
            stdlib.stdio.exit(0);
        } else if (std.mem.eql(u8, arg, "--version")) {
            tools_common.build_info.print(llama.api.llama.llama_version(), stdlib.stdio.stderr());
            stdlib.stdio.exit(0);
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
            params.n_split_tensors = stdlib.num.atoi(argv[arg_idx]);
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

/// Ports `split_params_parse` (gguf-split.cpp:183 @b1b770d90).
fn parse(argv: []const [:0]const u8, params: *Params, arena: std.mem.Allocator) !void {
    var err: ParseError = .{};
    parseEx(argv, params, &err, arena) catch |e| switch (e) {
        error.Invalid => {
            tools_common.io.eprintf("%.*s\n", .{ @as(c_int, @intCast(err.len)), &err.buf });
            printUsage(argv[0].ptr);
            stdlib.stdio.exit(1);
        },
        else => |other| return other,
    };
}

// -----------------------------------------------------------------------------
// Files, as the C++ streams

/// An output stream's `failbit` exception, which nothing catches.
fn terminate() noreturn {
    tools_common.io.eprintf("terminate called after throwing an instance of 'std::ios_base::failure'\n", .{});
    std.process.abort();
}

/// `std::ofstream fout(path, std::ios::binary)` with `failbit` exceptions:
/// a file that cannot be created terminates.
fn createOut(std_io: std.Io, path: []const u8) stdlib.file.Out {
    return stdlib.file.Out.create(std_io, path) catch terminate();
}

/// `fout.write(..)` with `failbit` exceptions: a failed write terminates.
fn writeAll(f: *stdlib.file.Out, bytes: []const u8) void {
    f.write(bytes) catch terminate();
}

/// Ports `zeros` (gguf-split.cpp:196 @b1b770d90).
fn zeros(f: *stdlib.file.Out, n: usize) void {
    f.zeros(n) catch terminate();
}

/// A path buffer as the `char *` the C passes to `printf`.
fn cstr(buf: *const [path_max:0]u8) [*:0]const u8 {
    return buf;
}

/// `GGML_PAD(x, GGUF_DEFAULT_ALIGNMENT)`.
fn pad(x: usize) usize {
    return stdlib.num.pad(usize, x, llama.impl.llama_c.GGUF_DEFAULT_ALIGNMENT);
}

// -----------------------------------------------------------------------------
// Split

/// Ports `split_strategy` (gguf-split.cpp:203 @b1b770d90).
const Strategy = struct {
    params: Params,
    std_io: std.Io,
    f_input: std.Io.File,
    ctx_gguf: *llama.impl.llama_c.gguf_context,
    ctx_meta: *llama.impl.llama_c.ggml_context,
    n_tensors: c_int,
    // one ctx_out per one output file
    ctx_outs: std.ArrayList(*llama.impl.llama_c.gguf_context) = .empty,
    // temporary buffer for reading in tensor data
    read_buf: std.ArrayList(u8) = .empty,
    allocator: std.mem.Allocator,

    // because we need to know list of tensors for each file in advance, we
    // will build all the ctx_out for all output splits
    fn init(allocator: std.mem.Allocator, std_io: std.Io, params: Params, f_input: std.Io.File, ctx_gguf: *llama.impl.llama_c.gguf_context, ctx_meta: *llama.impl.llama_c.ggml_context) !Strategy {
        var self: Strategy = .{
            .params = params,
            .std_io = std_io,
            .f_input = f_input,
            .ctx_gguf = ctx_gguf,
            .ctx_meta = ctx_meta,
            .n_tensors = @intCast(llama.impl.llama_c.gguf_get_n_tensors(ctx_gguf)),
            .allocator = allocator,
        };

        var i_split: c_int = -1;
        var ctx_out: ?*llama.impl.llama_c.gguf_context = null;

        // initialize ctx_out for the first split
        try self.newCtxOut(&i_split, &ctx_out, false);

        // skip first split if no_tensor_first_split is set
        if (params.no_tensor_first_split) try self.newCtxOut(&i_split, &ctx_out, true);

        // process tensors one by one
        var curr_tensors_size: usize = 0; // current size by counting only tensors size (without metadata)
        var i: c_int = 0;
        while (i < self.n_tensors) : (i += 1) {
            const t = llama.impl.llama_c.ggml_get_tensor(ctx_meta, llama.impl.llama_c.gguf_get_tensor_name(ctx_gguf, i));
            // calculate the "imaginary" size = the current size + next tensor size
            const n_bytes = pad(llama.impl.llama_c.ggml_nbytes(t));
            const next_tensors_size = curr_tensors_size + n_bytes;
            if (self.shouldSplit(i, next_tensors_size)) {
                try self.newCtxOut(&i_split, &ctx_out, false);
                curr_tensors_size = n_bytes;
            } else {
                curr_tensors_size = next_tensors_size;
            }
            llama.impl.llama_c.gguf_add_tensor(ctx_out.?, t);
        }

        // push the last ctx_out
        try self.ctx_outs.append(allocator, ctx_out.?);

        // set the correct n_split for all ctx_out
        for (self.ctx_outs.items) |ctx| llama.impl.llama_c.gguf_set_val_u16(ctx, tools_common.kv_split_count, @truncate(self.ctx_outs.items.len));
        return self;
    }

    /// The C++ constructor's `new_ctx_out` lambda (gguf-split.cpp:229 @b1b770d90).
    fn newCtxOut(self: *Strategy, i_split: *c_int, ctx_out: *?*llama.impl.llama_c.gguf_context, allow_no_tensors: bool) !void {
        i_split.* += 1;
        if (ctx_out.*) |prev| {
            if (llama.impl.llama_c.gguf_get_n_tensors(prev) == 0 and !allow_no_tensors) {
                tools_common.io.eprintf("error: one of splits have 0 tensors. Maybe size or tensors limit is too small\n", .{});
                stdlib.stdio.exit(1);
            }
            try self.ctx_outs.append(self.allocator, prev);
        }
        const out = llama.impl.llama_c.gguf_init_empty().?;
        ctx_out.* = out;
        // Save all metadata in first split only
        if (i_split.* == 0) llama.impl.llama_c.gguf_set_kv(out, self.ctx_gguf);
        llama.impl.llama_c.gguf_set_val_u16(out, tools_common.kv_split_no, @truncate(@as(c_uint, @bitCast(i_split.*))));
        llama.impl.llama_c.gguf_set_val_u16(out, tools_common.kv_split_count, 0); // placeholder
        llama.impl.llama_c.gguf_set_val_i32(out, tools_common.kv_split_tensors_count, self.n_tensors);
    }

    fn deinit(self: *Strategy) void {
        for (self.ctx_outs.items) |ctx_out| llama.impl.llama_c.gguf_free(ctx_out);
        self.ctx_outs.deinit(self.allocator);
        self.read_buf.deinit(self.allocator);
    }

    /// Ports `should_split` (gguf-split.cpp:287 @b1b770d90).
    fn shouldSplit(self: *const Strategy, i_tensor: c_int, next_size: usize) bool {
        return switch (self.params.mode) {
            // split by max size per file
            .size => next_size > self.params.n_bytes_split,
            // split by number of tensors per file
            .tensor => i_tensor > 0 and i_tensor < self.n_tensors and @rem(i_tensor, self.params.n_split_tensors) == 0,
            // should never happen
            .none => llama.impl.llama_c.ggml_abort("gguf-split.cpp", 296, "invalid mode"),
        };
    }

    /// Ports `print_info` (gguf-split.cpp:299 @b1b770d90).
    fn printInfo(self: *const Strategy) void {
        tools_common.io.printf("n_split: %zu\n", .{self.ctx_outs.items.len});
        var i_split: c_int = 0;
        for (self.ctx_outs.items) |ctx_out| {
            // re-calculate the real gguf size for each split (= metadata size + total size of all tensors)
            var total_size: usize = llama.impl.llama_c.gguf_get_meta_size(ctx_out);
            var i: i64 = 0;
            while (i < llama.impl.llama_c.gguf_get_n_tensors(ctx_out)) : (i += 1) {
                const t = llama.impl.llama_c.ggml_get_tensor(self.ctx_meta, llama.impl.llama_c.gguf_get_tensor_name(ctx_out, i));
                total_size += llama.impl.llama_c.ggml_nbytes(t);
            }
            total_size = total_size / 1000 / 1000; // convert to megabytes
            tools_common.io.printf("split %05d: n_tensors = %lld, total_size = %zuM\n", .{ i_split + 1, @as(c_longlong, llama.impl.llama_c.gguf_get_n_tensors(ctx_out)), total_size });
            i_split += 1;
        }
    }

    /// Ports `write` (gguf-split.cpp:315 @b1b770d90).
    fn write(self: *Strategy) !void {
        var i_split: c_int = 0;
        const n_split: c_int = @intCast(self.ctx_outs.items.len);
        for (self.ctx_outs.items) |ctx_out| {
            // construct file path
            var split_path: [path_max:0]u8 = @splat(0);
            _ = llama.api.llama.llama_split_path(&split_path, split_path.len, self.params.output.ptr, i_split, n_split);

            // open the output file
            tools_common.io.printf("Writing file %s ... ", .{cstr(&split_path)});
            stdlib.stdio.stdout().flush();
            var fout = createOut(self.std_io, std.mem.sliceTo(&split_path, 0));

            // write metadata
            const data = try self.allocator.alloc(u8, llama.impl.llama_c.gguf_get_meta_size(ctx_out));
            defer self.allocator.free(data);
            llama.impl.llama_c.gguf_get_meta_data(ctx_out, data.ptr);
            writeAll(&fout, data);

            // write tensors
            var i: i64 = 0;
            while (i < llama.impl.llama_c.gguf_get_n_tensors(ctx_out)) : (i += 1) {
                // read tensor meta and prepare buffer
                const t_name = llama.impl.llama_c.gguf_get_tensor_name(ctx_out, i);
                const t = llama.impl.llama_c.ggml_get_tensor(self.ctx_meta, t_name);
                const n_bytes = llama.impl.llama_c.ggml_nbytes(t);
                try self.read_buf.resize(self.allocator, n_bytes);

                // calculate offset
                const i_tensor_in = llama.impl.llama_c.gguf_find_tensor(self.ctx_gguf, t_name); // idx of tensor in the input file
                const offset = llama.impl.llama_c.gguf_get_data_offset(self.ctx_gguf) + llama.impl.llama_c.gguf_get_tensor_offset(self.ctx_gguf, i_tensor_in);

                // copy tensor from input to output file
                try self.copyFileToFile(&fout, offset, n_bytes);
                zeros(&fout, pad(n_bytes) - n_bytes);
            }

            tools_common.io.printf("done\n", .{});
            // close the file
            fout.close(self.std_io);
            i_split += 1;
        }
    }

    /// Ports `copy_file_to_file` (gguf-split.cpp:358 @b1b770d90).
    fn copyFileToFile(self: *Strategy, f_out: *stdlib.file.Out, in_offset: usize, len: usize) !void {
        if (self.read_buf.items.len < len) try self.read_buf.resize(self.allocator, len);
        if (!stdlib.file.readAt(self.std_io, self.f_input, self.read_buf.items[0..len], in_offset)) {
            tools_common.io.eprintf("%s: failed to read %zu bytes at offset %zu, input file may be truncated\n", .{ "copy_file_to_file", len, in_offset });
            stdlib.stdio.exit(1);
        }
        writeAll(f_out, self.read_buf.items[0..len]);
    }
};

/// Ports `gguf_split` (gguf-split.cpp:373 @b1b770d90).
fn ggufSplit(allocator: std.mem.Allocator, std_io: std.Io, split_params: Params) !void {
    var ctx_meta: ?*llama.impl.llama_c.ggml_context = null;

    const params: llama.impl.llama_c.gguf_init_params = .{ .no_alloc = true, .ctx = &ctx_meta };

    const f_input = stdlib.file.openIn(std_io, split_params.input) catch {
        tools_common.io.eprintf("%s:  failed to open input GGUF from %s\n", .{ "gguf_split", split_params.input.ptr });
        stdlib.stdio.exit(1);
    };

    const ctx_gguf = llama.impl.llama_c.gguf_init_from_file(split_params.input.ptr, params) orelse {
        tools_common.io.eprintf("%s:  failed to load input GGUF from %s\n", .{ "gguf_split", split_params.input.ptr });
        stdlib.stdio.exit(1);
    };

    // prepare the strategy
    var strategy = try Strategy.init(allocator, std_io, split_params, f_input, ctx_gguf, ctx_meta.?);
    defer strategy.deinit();
    const n_split: c_int = @intCast(strategy.ctx_outs.items.len);
    strategy.printInfo();

    if (!split_params.dry_run) {
        // write all output splits
        try strategy.write();
    }

    // done, clean up
    llama.impl.llama_c.gguf_free(ctx_gguf);
    f_input.close(std_io);

    tools_common.io.eprintf("%s: %d gguf split written with a total of %d tensors.\n", .{ "gguf_split", n_split, strategy.n_tensors });
}

/// Ports `gguf_merge` (gguf-split.cpp:411 @b1b770d90).
fn ggufMerge(allocator: std.mem.Allocator, std_io: std.Io, split_params: Params) !void {
    const func = "gguf_merge";
    tools_common.io.eprintf("%s: %s -> %s\n", .{ func, split_params.input.ptr, split_params.output.ptr });
    var n_split: c_int = 1;
    var total_tensors: c_int = 0;

    // avoid overwriting existing output file
    if (stdlib.file.openIn(std_io, split_params.output)) |existing| {
        existing.close(std_io);
        tools_common.io.eprintf("%s: output file %s already exists\n", .{ func, split_params.output.ptr });
        stdlib.stdio.exit(1);
    } else |_| {}

    const ctx_out = llama.impl.llama_c.gguf_init_empty().?;

    var read_data: std.ArrayList(u8) = .empty;
    defer read_data.deinit(allocator);
    var ctx_metas: std.ArrayList(*llama.impl.llama_c.ggml_context) = .empty;
    defer ctx_metas.deinit(allocator);
    var ctx_ggufs: std.ArrayList(*llama.impl.llama_c.gguf_context) = .empty;
    defer ctx_ggufs.deinit(allocator);

    var split_path: [path_max:0]u8 = @splat(0);
    const in_len = @min(split_params.input.len, split_path.len - 1);
    @memcpy(split_path[0..in_len], split_params.input[0..in_len]);
    var split_prefix: [path_max:0]u8 = @splat(0);

    // First pass to find KV and tensors metadata
    var i_split: c_int = 0;
    while (i_split < n_split) : (i_split += 1) {
        var ctx_meta: ?*llama.impl.llama_c.ggml_context = null;

        const params: llama.impl.llama_c.gguf_init_params = .{ .no_alloc = true, .ctx = &ctx_meta };

        if (i_split > 0) _ = llama.api.llama.llama_split_path(&split_path, split_path.len, &split_prefix, i_split, n_split);
        tools_common.io.eprintf("%s: reading metadata %s ...", .{ func, cstr(&split_path) });

        const ctx_gguf = llama.impl.llama_c.gguf_init_from_file(&split_path, params) orelse {
            tools_common.io.eprintf("\n%s:  failed to load input GGUF from %s\n", .{ func, split_params.input.ptr });
            stdlib.stdio.exit(1);
        };
        try ctx_ggufs.append(allocator, ctx_gguf);
        try ctx_metas.append(allocator, ctx_meta.?);

        // Verify that this split belongs to the same set as the first one
        if (i_split > 0) {
            const key_split_no = llama.impl.llama_c.gguf_find_key(ctx_gguf, tools_common.kv_split_no);
            const key_split_count = llama.impl.llama_c.gguf_find_key(ctx_gguf, tools_common.kv_split_count);
            if (key_split_no < 0 or llama.impl.llama_c.gguf_get_val_u16(ctx_gguf, key_split_no) != i_split or
                key_split_count < 0 or llama.impl.llama_c.gguf_get_val_u16(ctx_gguf, key_split_count) != n_split)
            {
                tools_common.io.eprintf("\n%s: %s is not split %d of %d\n", .{ func, cstr(&split_path), i_split + 1, n_split });
                stdlib.stdio.exit(1);
            }
        }

        if (i_split == 0) {
            const key_n_split = llama.impl.llama_c.gguf_find_key(ctx_gguf, tools_common.kv_split_count);
            if (key_n_split < 0) {
                tools_common.io.eprintf("\n%s: input file does not contain %s metadata\n", .{ func, tools_common.kv_split_count.ptr });
                llama.impl.llama_c.gguf_free(ctx_gguf);
                llama.impl.llama_c.ggml_free(ctx_meta);
                llama.impl.llama_c.gguf_free(ctx_out);
                stdlib.stdio.exit(1);
            }

            n_split = llama.impl.llama_c.gguf_get_val_u16(ctx_gguf, key_n_split);
            if (n_split < 1) {
                tools_common.io.eprintf("\n%s: input file does not contain a valid split count %d\n", .{ func, n_split });
                llama.impl.llama_c.gguf_free(ctx_gguf);
                llama.impl.llama_c.ggml_free(ctx_meta);
                llama.impl.llama_c.gguf_free(ctx_out);
                stdlib.stdio.exit(1);
            }

            // Verify the file naming and extract split_prefix
            if (llama.api.llama.llama_split_prefix(&split_prefix, split_prefix.len, &split_path, i_split, n_split) == 0) {
                tools_common.io.eprintf("\n%s: unexpected input file name: %s i_split=%d n_split=%d\n", .{ func, cstr(&split_path), i_split, n_split });
                llama.impl.llama_c.gguf_free(ctx_gguf);
                llama.impl.llama_c.ggml_free(ctx_meta);
                llama.impl.llama_c.gguf_free(ctx_out);
                stdlib.stdio.exit(1);
            }

            // Do not trigger merge if we try to merge again the output
            llama.impl.llama_c.gguf_set_val_u16(ctx_gguf, tools_common.kv_split_count, 0);

            // Set metadata from the first split
            llama.impl.llama_c.gguf_set_kv(ctx_out, ctx_gguf);
        }

        const n_tensors = llama.impl.llama_c.gguf_get_n_tensors(ctx_gguf);
        var i_tensor: i64 = 0;
        while (i_tensor < n_tensors) : (i_tensor += 1) {
            const t_name = llama.impl.llama_c.gguf_get_tensor_name(ctx_gguf, i_tensor);
            const t = llama.impl.llama_c.ggml_get_tensor(ctx_meta, t_name);
            llama.impl.llama_c.gguf_add_tensor(ctx_out, t);
        }
        total_tensors += @intCast(n_tensors);

        tools_common.io.eprintf("\x1b[3Ddone\n", .{});
    }

    // Verify that no tensor is missing
    const key_n_tensors = llama.impl.llama_c.gguf_find_key(ctx_ggufs.items[0], tools_common.kv_split_tensors_count);
    if (key_n_tensors < 0 or llama.impl.llama_c.gguf_get_val_i32(ctx_ggufs.items[0], key_n_tensors) != total_tensors) {
        tools_common.io.eprintf("%s: splits contain %d tensors, which does not match %s\n", .{ func, total_tensors, tools_common.kv_split_tensors_count.ptr });
        stdlib.stdio.exit(1);
    }

    var fout: stdlib.file.Out = undefined;
    if (!split_params.dry_run) {
        fout = createOut(std_io, split_params.output);
        // placeholder for the meta data
        const meta_size = llama.impl.llama_c.gguf_get_meta_size(ctx_out);
        zeros(&fout, meta_size);
    }

    // Write tensors data
    var merge_error = false;
    i_split = 0;
    while (i_split < n_split) : (i_split += 1) {
        _ = llama.api.llama.llama_split_path(&split_path, split_path.len, &split_prefix, i_split, n_split);
        const f_input = stdlib.file.openIn(std_io, std.mem.sliceTo(&split_path, 0)) catch {
            tools_common.io.eprintf("%s:  failed to open input GGUF from %s\n", .{ func, cstr(&split_path) });
            for (ctx_ggufs.items, ctx_metas.items) |g, m| {
                llama.impl.llama_c.gguf_free(g);
                llama.impl.llama_c.ggml_free(m);
            }
            llama.impl.llama_c.gguf_free(ctx_out);
            if (!split_params.dry_run) fout.close(std_io);
            stdlib.stdio.exit(1);
        };
        tools_common.io.eprintf("%s: writing tensors %s ...", .{ func, cstr(&split_path) });

        const ctx_gguf = ctx_ggufs.items[@intCast(i_split)];
        const ctx_meta = ctx_metas.items[@intCast(i_split)];

        const n_tensors = llama.impl.llama_c.gguf_get_n_tensors(ctx_gguf);
        var i_tensor: i64 = 0;
        while (i_tensor < n_tensors) : (i_tensor += 1) {
            const t_name = llama.impl.llama_c.gguf_get_tensor_name(ctx_gguf, i_tensor);
            const t = llama.impl.llama_c.ggml_get_tensor(ctx_meta, t_name);

            const n_bytes = llama.impl.llama_c.ggml_nbytes(t);

            if (read_data.items.len < n_bytes) try read_data.resize(allocator, n_bytes);

            const offset = llama.impl.llama_c.gguf_get_data_offset(ctx_gguf) + llama.impl.llama_c.gguf_get_tensor_offset(ctx_gguf, i_tensor);
            if (!stdlib.file.readAt(std_io, f_input, read_data.items[0..n_bytes], offset)) {
                tools_common.io.eprintf("%s: failed to read %zu bytes at offset %zu, input file may be truncated\n", .{ func, n_bytes, offset });
                stdlib.stdio.exit(1);
            }
            if (!split_params.dry_run) {
                // write tensor data + padding
                writeAll(&fout, read_data.items[0..n_bytes]);
                zeros(&fout, pad(n_bytes) - n_bytes);
            }
        }

        llama.impl.llama_c.gguf_free(ctx_gguf);
        llama.impl.llama_c.ggml_free(ctx_meta);
        f_input.close(std_io);
        tools_common.io.eprintf("\x1b[3Ddone\n", .{});

        if (!split_params.dry_run and split_params.delete_splits) {
            if (stdlib.file.remove(std_io, std.mem.sliceTo(&split_path, 0))) {
                tools_common.io.eprintf("%s: deleted file %s\n", .{ func, cstr(&split_path) });
            } else {
                merge_error = true;
                tools_common.io.eprintf("error: failed to delete %s\n", .{cstr(&split_path)});
            }
        }
    }

    if (!split_params.dry_run) {
        // go back to beginning of file and write the updated metadata
        fout.seek(0) catch terminate();
        const data = try allocator.alloc(u8, llama.impl.llama_c.gguf_get_meta_size(ctx_out));
        defer allocator.free(data);
        llama.impl.llama_c.gguf_get_meta_data(ctx_out, data.ptr);
        writeAll(&fout, data);
        fout.close(std_io);
    }
    llama.impl.llama_c.gguf_free(ctx_out);

    tools_common.io.eprintf("%s: %s merged from %d split with %d tensors.\n", .{ func, split_params.output.ptr, n_split, total_tensors });

    if (merge_error) stdlib.stdio.exit(1);
}

/// Ports `main` (gguf-split.cpp:620 @b1b770d90).
pub fn main(init: std.process.Init) !void {
    stdlib.stdio.bufferAsC();
    stdlib.env.set(init.environ_map);
    defer stdlib.stdio.flushAll();
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);

    var params: Params = .{};
    try parse(argv, &params, arena);

    switch (params.operation) {
        .split => try ggufSplit(init.gpa, init.io, params),
        .merge => try ggufMerge(init.gpa, init.io, params),
        else => {
            printUsage(argv[0].ptr);
            stdlib.stdio.exit(1);
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
