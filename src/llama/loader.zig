//! Opening a model: its metadata, its tensors' whereabouts, its file type.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-model-loader.cpp` -- `llama_file_version_name`,
//!   `llama_ftype_name`, `llama_get_list_splits`, the constructor of
//!   `llama_model_loader`, `get_arch_name`, `get_arch`, `ftype_name` and
//!   `print_info`
//! - `llama.cpp/src/llama-model-loader.h` -- `llama_fver`,
//!   `llama_tensor_weight` and `weight_name_comparer`
//! - `llama.cpp/src/llama-impl.cpp` -- `replace_all`, `gguf_data_to_str`
//!   and `gguf_kv_to_str`
//! - `llama.cpp/src/llama-mmap.cpp` -- the opening half of `llama_file`
//!
//! All at v0.3.0 (`c1d0e7a00`). The typed getters the constructor and the
//! model use are `gguf_meta.zig`'s.
//!
//! # Scope
//!
//! The metadata half of `llama_model_loader`: everything up to and
//! including `print_info`, which is what loading hyperparameters needs.
//! Creating tensors, mapping and reading their data, and `llama_file`'s
//! reads come with the tensor loading that uses them.
//!
//! # The weights' order is the C++ map's
//!
//! `weights_map` is a `std::map` under `weight_name_comparer`: by the
//! block index `sscanf` reads from `blk.%d.`, then by name. The file type
//! is guessed from it in that order, and the first type to reach the
//! highest count wins a tie, so the order is part of the answer. `Weight`s
//! are sorted the same way here, and the block index is read by libc's own
//! `sscanf` so its edge cases are the C++'s.

const std = @import("std");
const impl = @import("impl.zig");
const arch = @import("arch.zig");
const gguf_meta = @import("gguf_meta.zig");

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;

extern "c" fn strerror(errnum: c_int) [*:0]const u8;
/// `errno`'s location: `__error` on Darwin, `__errno_location` on glibc.
const errnoPtr = std.c._errno;
extern "c" fn atoi(s: [*:0]const u8) c_int;

const MiB: usize = 1024 * 1024;
const GiB: usize = 1024 * MiB;

/// Mirrors `llama_fver` (llama-model-loader.h:23 @c1d0e7a00).
pub const FileVersion = enum(u32) {
    v1 = 1,
    v2 = 2,
    v3 = 3,
    _,
};

/// Ports `llama_file_version_name` (llama-model-loader.cpp:21 @c1d0e7a00).
pub fn fileVersionName(version: FileVersion) [:0]const u8 {
    return switch (version) {
        .v1 => "GGUF V1 (support until nov 2023)",
        .v2 => "GGUF V2",
        .v3 => "GGUF V3 (latest)",
        _ => "unknown",
    };
}

/// Ports `llama_ftype_name` (llama-model-loader.cpp:33 @c1d0e7a00): the
/// name, with `(guessed) ` in front when `LLAMA_FTYPE_GUESSED` is set.
pub fn ftypeName(ftype: c_uint) [:0]const u8 {
    const prefixed: [:0]const u8 = switch (ftype & ~@as(c_uint, c.LLAMA_FTYPE_GUESSED)) {
        c.LLAMA_FTYPE_ALL_F32 => "(guessed) all F32",
        c.LLAMA_FTYPE_MOSTLY_F16 => "(guessed) F16",
        c.LLAMA_FTYPE_MOSTLY_BF16 => "(guessed) BF16",
        c.LLAMA_FTYPE_MOSTLY_Q1_0 => "(guessed) Q1_0",
        c.LLAMA_FTYPE_MOSTLY_Q2_0 => "(guessed) Q2_0",
        c.LLAMA_FTYPE_MOSTLY_Q4_0 => "(guessed) Q4_0",
        c.LLAMA_FTYPE_MOSTLY_Q4_1 => "(guessed) Q4_1",
        c.LLAMA_FTYPE_MOSTLY_Q5_0 => "(guessed) Q5_0",
        c.LLAMA_FTYPE_MOSTLY_Q5_1 => "(guessed) Q5_1",
        c.LLAMA_FTYPE_MOSTLY_Q8_0 => "(guessed) Q8_0",
        c.LLAMA_FTYPE_MOSTLY_MXFP4_MOE => "(guessed) MXFP4 MoE",
        c.LLAMA_FTYPE_MOSTLY_NVFP4 => "(guessed) NVFP4",
        c.LLAMA_FTYPE_MOSTLY_Q2_K => "(guessed) Q2_K - Medium",
        c.LLAMA_FTYPE_MOSTLY_Q2_K_S => "(guessed) Q2_K - Small",
        c.LLAMA_FTYPE_MOSTLY_Q3_K_S => "(guessed) Q3_K - Small",
        c.LLAMA_FTYPE_MOSTLY_Q3_K_M => "(guessed) Q3_K - Medium",
        c.LLAMA_FTYPE_MOSTLY_Q3_K_L => "(guessed) Q3_K - Large",
        c.LLAMA_FTYPE_MOSTLY_Q4_K_S => "(guessed) Q4_K - Small",
        c.LLAMA_FTYPE_MOSTLY_Q4_K_M => "(guessed) Q4_K - Medium",
        c.LLAMA_FTYPE_MOSTLY_Q5_K_S => "(guessed) Q5_K - Small",
        c.LLAMA_FTYPE_MOSTLY_Q5_K_M => "(guessed) Q5_K - Medium",
        c.LLAMA_FTYPE_MOSTLY_Q6_K => "(guessed) Q6_K",
        c.LLAMA_FTYPE_MOSTLY_TQ1_0 => "(guessed) TQ1_0 - 1.69 bpw ternary",
        c.LLAMA_FTYPE_MOSTLY_TQ2_0 => "(guessed) TQ2_0 - 2.06 bpw ternary",
        c.LLAMA_FTYPE_MOSTLY_IQ2_XXS => "(guessed) IQ2_XXS - 2.0625 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ2_XS => "(guessed) IQ2_XS - 2.3125 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ2_S => "(guessed) IQ2_S - 2.5 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ2_M => "(guessed) IQ2_M - 2.7 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ3_XS => "(guessed) IQ3_XS - 3.3 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ3_XXS => "(guessed) IQ3_XXS - 3.0625 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ1_S => "(guessed) IQ1_S - 1.5625 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ1_M => "(guessed) IQ1_M - 1.75 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ4_NL => "(guessed) IQ4_NL - 4.5 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ4_XS => "(guessed) IQ4_XS - 4.25 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ3_S => "(guessed) IQ3_S - 3.4375 bpw",
        c.LLAMA_FTYPE_MOSTLY_IQ3_M => "(guessed) IQ3_S mix - 3.66 bpw",
        else => "(guessed) unknown, may not work",
    };
    const guessed_prefix_len = "(guessed) ".len;
    return if (ftype & c.LLAMA_FTYPE_GUESSED != 0) prefixed else prefixed[guessed_prefix_len..];
}

// -----------------------------------------------------------------------------
// Metadata as text

/// Ports `replace_all` (llama-impl.cpp:69 @c1d0e7a00).
///
/// Parameters:
/// - `s`: the string, rewritten in place.
/// - `search`: what to replace; nothing happens when it is empty.
/// - `replace`: what replaces it.
///
/// Return: nothing.
pub fn replaceAll(s: *std.ArrayList(u8), search: []const u8, replace: []const u8) void {
    if (search.len == 0) return;
    var builder: std.ArrayList(u8) = .empty;
    builder.ensureTotalCapacity(allocator, s.items.len) catch oom();
    var last_pos: usize = 0;
    while (std.mem.findPos(u8, s.items, last_pos, search)) |pos| {
        builder.appendSlice(allocator, s.items[last_pos..pos]) catch oom();
        builder.appendSlice(allocator, replace) catch oom();
        last_pos = pos + search.len;
    }
    builder.appendSlice(allocator, s.items[last_pos..]) catch oom();
    s.deinit(allocator);
    s.* = builder;
}

fn oom() noreturn {
    impl.abort("llama-model-loader.cpp", 0, "out of memory");
}

fn appendPrint(out: *std.ArrayList(u8), comptime fmt: []const u8, args: anytype) void {
    out.print(allocator, fmt, args) catch oom();
}

/// `snprintf` with a C format, appended; for the conversions whose exact
/// digits only libc can promise, `%f` above all.
fn appendC(out: *std.ArrayList(u8), comptime fmt: [:0]const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const n = @call(.auto, c.snprintf, .{ &buf, buf.len, fmt.ptr } ++ args);
    out.appendSlice(allocator, buf[0..@min(@as(usize, @intCast(n)), buf.len - 1)]) catch oom();
}

/// Ports `gguf_data_to_str` (llama-impl.cpp:119 @c1d0e7a00): one element
/// as `std::to_string` writes it, so a float has six decimals.
fn dataToStr(out: *std.ArrayList(u8), ty: c.enum_gguf_type, data: ?*const anyopaque, i: usize) void {
    const p: [*]const u8 = @ptrCast(data.?);
    switch (ty) {
        c.GGUF_TYPE_UINT8 => appendPrint(out, "{d}", .{p[i]}),
        c.GGUF_TYPE_INT8 => appendPrint(out, "{d}", .{@as([*]const i8, @ptrCast(p))[i]}),
        c.GGUF_TYPE_UINT16 => appendPrint(out, "{d}", .{@as([*]align(1) const u16, @ptrCast(p))[i]}),
        c.GGUF_TYPE_INT16 => appendPrint(out, "{d}", .{@as([*]align(1) const i16, @ptrCast(p))[i]}),
        c.GGUF_TYPE_UINT32 => appendPrint(out, "{d}", .{@as([*]align(1) const u32, @ptrCast(p))[i]}),
        c.GGUF_TYPE_INT32 => appendPrint(out, "{d}", .{@as([*]align(1) const i32, @ptrCast(p))[i]}),
        c.GGUF_TYPE_UINT64 => appendPrint(out, "{d}", .{@as([*]align(1) const u64, @ptrCast(p))[i]}),
        c.GGUF_TYPE_INT64 => appendPrint(out, "{d}", .{@as([*]align(1) const i64, @ptrCast(p))[i]}),
        c.GGUF_TYPE_FLOAT32 => appendC(out, "%f", .{@as(f64, @as([*]align(1) const f32, @ptrCast(p))[i])}),
        c.GGUF_TYPE_FLOAT64 => appendC(out, "%f", .{@as([*]align(1) const f64, @ptrCast(p))[i]}),
        c.GGUF_TYPE_BOOL => appendPrint(out, "{s}", .{if (@as([*]const i8, @ptrCast(p))[i] != 0) "true" else "false"}),
        else => appendPrint(out, "unknown type {d}", .{ty}),
    }
}

/// Ports `gguf_kv_to_str` (llama-impl.cpp:136 @c1d0e7a00).
///
/// Parameters:
/// - `ctx`: the metadata.
/// - `i`: the key's index.
/// - `out`: receives the text, appended.
///
/// Return: nothing.
pub fn kvToStr(ctx: *const c.gguf_context, i: i64, out: *std.ArrayList(u8)) void {
    const ty = c.gguf_get_kv_type(ctx, i);
    switch (ty) {
        c.GGUF_TYPE_STRING => out.appendSlice(allocator, gguf_meta.cString(c.gguf_get_val_str(ctx, i))) catch oom(),
        c.GGUF_TYPE_ARRAY => {
            const arr_type = c.gguf_get_arr_type(ctx, i);
            // `int arr_n`: the C++ narrows the length.
            const arr_n: c_int = @truncate(@as(i64, @bitCast(c.gguf_get_arr_n(ctx, i))));
            const data = if (arr_type == c.GGUF_TYPE_STRING) null else c.gguf_get_arr_data(ctx, i);
            out.append(allocator, '[') catch oom();
            var j: c_int = 0;
            while (j < arr_n) : (j += 1) {
                if (arr_type == c.GGUF_TYPE_STRING) {
                    var val: std.ArrayList(u8) = .empty;
                    defer val.deinit(allocator);
                    val.appendSlice(allocator, gguf_meta.cString(c.gguf_get_arr_str(ctx, i, @intCast(j)))) catch oom();
                    replaceAll(&val, "\\", "\\\\");
                    replaceAll(&val, "\"", "\\\"");
                    out.append(allocator, '"') catch oom();
                    out.appendSlice(allocator, val.items) catch oom();
                    out.append(allocator, '"') catch oom();
                } else if (arr_type == c.GGUF_TYPE_ARRAY) {
                    out.appendSlice(allocator, "???") catch oom();
                } else {
                    dataToStr(out, arr_type, data, @intCast(j));
                }
                if (j < arr_n - 1) out.appendSlice(allocator, ", ") catch oom();
            }
            out.append(allocator, ']') catch oom();
        },
        else => dataToStr(out, ty, c.gguf_get_val_data(ctx, i), 0),
    }
}

// -----------------------------------------------------------------------------
// Files

/// The opening half of `llama_file`: the handle and its size.
pub const File = struct {
    fp: *c.FILE,
    owns_fp: bool,
    size: usize,

    /// Ports `llama_file::impl`'s `init_fp` (llama-mmap.cpp:218 @c1d0e7a00):
    /// open, then seek to the end for the size.
    ///
    /// Parameters:
    /// - `fname`: the path.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the file, or `error.Thrown` when it cannot be opened.
    pub fn open(fname: [:0]const u8, exc: *Exception) Thrown!File {
        const fp = c.ggml_fopen(fname.ptr, "rb") orelse {
            return exc.throw(.runtime_error, "failed to open {s}: {s}", .{ fname, std.mem.span(strerror(errnoPtr().*)) });
        };
        var f: File = .{ .fp = fp, .owns_fp = true, .size = 0 };
        f.size = try f.measure(exc);
        return f;
    }

    /// Ports the `FILE *` constructor, `impl` (llama-mmap.cpp:228
    /// @c1d0e7a00): borrowed, measured the same way.
    pub fn borrow(fp: *c.FILE, exc: *Exception) Thrown!File {
        var f: File = .{ .fp = fp, .owns_fp = false, .size = 0 };
        f.size = try f.measure(exc);
        return f;
    }

    fn measure(self: *File, exc: *Exception) Thrown!usize {
        if (c.fseeko(self.fp, 0, c.SEEK_END) == -1)
            return exc.throw(.runtime_error, "seek error: {s}", .{std.mem.span(strerror(errnoPtr().*))});
        const size = c.ftello(self.fp);
        if (size == -1) return exc.throw(.runtime_error, "ftell error: {s}", .{std.mem.span(strerror(errnoPtr().*))});
        if (c.fseeko(self.fp, 0, c.SEEK_SET) == -1)
            return exc.throw(.runtime_error, "seek error: {s}", .{std.mem.span(strerror(errnoPtr().*))});
        return @intCast(size);
    }

    /// Return: nothing.
    pub fn close(self: *File) void {
        if (self.owns_fp) _ = c.fclose(self.fp);
    }
};

// -----------------------------------------------------------------------------
// Weights

/// Ports `llama_tensor_weight` (llama-model-loader.h:33 @c1d0e7a00).
pub const Weight = struct {
    /// Which file the data is in.
    idx: u16,
    /// The data's offset in that file.
    offs: usize,
    tensor: *c.ggml_tensor,
    /// `sscanf(name, "blk.%d.", &layer)`, -1 where it reads nothing.
    layer: c_int,

    /// Ports the constructor of `llama_tensor_weight` (llama-model-loader.h:39
    /// @c1d0e7a00): the tensor's offset, checked against the file's size.
    fn init(file: *const File, idx: u16, gguf_ctx: *const c.gguf_context, tensor: *c.ggml_tensor, exc: *Exception) Thrown!Weight {
        const tname = gguf_meta.cString(c.ggml_get_name(tensor));
        const tensor_idx = c.gguf_find_tensor(gguf_ctx, c.ggml_get_name(tensor));
        if (tensor_idx < 0) return exc.throw(.runtime_error, "tensor '{s}' not found in the model", .{tname});
        const offs = c.gguf_get_data_offset(gguf_ctx) +% c.gguf_get_tensor_offset(gguf_ctx, tensor_idx);
        const nbytes = c.ggml_nbytes(tensor);
        if (offs +% nbytes < offs or offs +% nbytes > file.size) {
            return exc.throw(.runtime_error, "tensor '{s}' data is not within the file bounds, model is corrupted or incomplete", .{tname});
        }
        return .{ .idx = idx, .offs = offs, .tensor = tensor, .layer = blockOf(c.ggml_get_name(tensor)) };
    }

    fn name(self: *const Weight) []const u8 {
        return gguf_meta.cString(c.ggml_get_name(self.tensor));
    }

    /// Ports `weight_name_comparer` (llama-model-loader.h:53 @c1d0e7a00).
    fn lessThan(_: void, a: Weight, b: Weight) bool {
        if (a.layer != b.layer) return a.layer < b.layer;
        return std.mem.order(u8, a.name(), b.name()) == .lt;
    }
};

fn blockOf(name: [*c]const u8) c_int {
    var layer: c_int = -1;
    _ = c.sscanf(name, "blk.%d.", &layer);
    return layer;
}

// -----------------------------------------------------------------------------
// The loader

/// The metadata half of `llama_model_loader`.
pub const Loader = struct {
    n_kv: c_int = 0,
    n_tensors: c_int = 0,
    n_created: c_int = 0,
    n_elements: u64 = 0,
    n_bytes: usize = 0,

    use_mmap: bool = false,
    use_direct_io: bool = false,
    check_tensors: bool = false,
    no_alloc: bool = false,
    load_mtp: bool = false,

    files: std.ArrayList(File) = .empty,
    ftype: c_uint = 0,
    fver: FileVersion = .v3,

    /// `weights_map`, in `weight_name_comparer` order once construction is
    /// done.
    weights: std.ArrayList(Weight) = .empty,

    /// The metadata with the caller's overrides, for the typed getters.
    meta: gguf_meta.Meta,
    /// `metadata_ptr`: the metadata when this loader opened it.
    owned: ?*c.gguf_context = null,
    /// The tensor-metadata contexts, one per file.
    contexts: std.ArrayList(*c.ggml_context) = .empty,

    arch_name: std.ArrayList(u8) = .empty,

    /// The type counts `guessFtype` leaves for `dumpMetadata`.
    n_type_counts: [c.GGML_TYPE_COUNT]u32 = @splat(0),

    /// Where the model comes from: exactly one of these.
    pub const Source = union(enum) {
        metadata: *c.gguf_context,
        path: [:0]const u8,
        file: *c.FILE,
    };

    /// Ports the constructor of `llama_model_loader` (llama-model-loader.cpp:527
    /// @c1d0e7a00), up to `print_info`.
    ///
    /// Parameters:
    /// - `source`: the metadata, a path, or an open file.
    /// - `splits`: the split paths when they do not follow the naming
    ///   scheme, else empty; filled in when needed. Borrowed.
    /// - `load_mode`: the load mode, for `use_mmap` and `use_direct_io`.
    /// - `check_tensors`, `no_alloc`, `load_mtp`: stored.
    /// - `overrides`: the caller's metadata overrides, or null.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the loader, or `error.Thrown`; free it with `deinit`.
    pub fn init(
        source: Source,
        splits: *std.ArrayList([:0]const u8),
        load_mode: c.enum_llama_load_mode,
        check_tensors: bool,
        no_alloc: bool,
        load_mtp: bool,
        overrides: ?[*]const c.llama_model_kv_override,
        exc: *Exception,
    ) Thrown!Loader {
        const trace: c_int = if (std.c.getenv("LLAMA_TRACE")) |t| atoi(t) else 0;

        var self: Loader = .{ .meta = gguf_meta.Meta.init(undefined, .unknown, overrides) };
        errdefer self.deinit();

        self.use_mmap = load_mode == c.LLAMA_LOAD_MODE_MMAP or load_mode == c.LLAMA_LOAD_MODE_MMAP_MLOCK or
            load_mode == c.LLAMA_LOAD_MODE_AUTO;
        self.use_direct_io = load_mode == c.LLAMA_LOAD_MODE_DIRECT_IO;

        const fname: []const u8 = switch (source) {
            .path => |p| p,
            else => "",
        };

        switch (source) {
            .path => |path| {
                var ctx: ?*c.ggml_context = null;
                self.owned = c.gguf_init_from_file(path.ptr, .{ .no_alloc = true, .ctx = &ctx }) orelse
                    return exc.throw(.runtime_error, "llama_model_loader: failed to load model from {s}", .{path});
                self.meta.ctx = self.owned.?;
                try self.readArch(exc);

                self.files.append(allocator, try File.open(path, exc)) catch oom();
                self.contexts.append(allocator, ctx.?) catch oom();
                try self.addWeights(ctx.?, 0, self.meta.ctx, exc);

                const n_split = (try self.meta.getKey(u16, .split_count, false, exc)) orelse 0;
                if (n_split > 1) try self.loadSplits(path, splits, n_split, trace, exc);
            },
            .file => |fp| {
                var ctx: ?*c.ggml_context = null;
                self.owned = c.gguf_init_from_file_ptr(fp, .{ .no_alloc = true, .ctx = &ctx }) orelse
                    return exc.throw(.runtime_error, "llama_model_loader: failed to load model from file pointer", .{});
                self.meta.ctx = self.owned.?;
                try self.readArch(exc);

                self.files.append(allocator, try File.borrow(fp, exc)) catch oom();
                self.contexts.append(allocator, ctx.?) catch oom();
                try self.addWeights(ctx.?, 0, self.meta.ctx, exc);
            },
            .metadata => |m| {
                self.meta.ctx = m;
                try self.readArch(exc);
            },
        }

        std.sort.pdq(Weight, self.weights.items, {}, Weight.lessThan);

        self.n_kv = @truncate(c.gguf_get_n_kv(self.meta.ctx));
        self.n_tensors = @intCast(self.weights.items.len);
        self.fver = @fromBackingInt(@intCast(c.gguf_get_version(self.meta.ctx)));

        const shown: []const u8 = if (fname.len == 0) "(file*)" else fname;
        impl.logInfo("%s: loaded meta data with %d key-value pairs and %d tensors from %.*s (version %s)\n", .{
            "llama_model_loader", self.n_kv, self.n_tensors, @as(c_int, @intCast(shown.len)), shown.ptr, fileVersionName(self.fver).ptr,
        });

        try self.guessFtype(trace, exc);
        self.dumpMetadata();

        // `llama_mmap::SUPPORTED` holds on every target this builds for.
        self.check_tensors = check_tensors;
        self.no_alloc = no_alloc;
        self.load_mtp = load_mtp;
        return self;
    }

    /// `get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false)`,
    /// then `llm_kv = LLM_KV(llm_arch_from_string(arch_name))`.
    fn readArch(self: *Loader, exc: *Exception) Thrown!void {
        if (try self.meta.getKey([]const u8, .general_architecture, false, exc)) |name| {
            self.arch_name.clearRetainingCapacity();
            self.arch_name.appendSlice(allocator, name) catch oom();
        }
        self.meta.arch = arch.archFromString(self.arch_name.items);
    }

    fn addWeights(self: *Loader, ctx: *c.ggml_context, idx: u16, gguf_ctx: *const c.gguf_context, exc: *Exception) Thrown!void {
        var cur = c.ggml_get_first_tensor(ctx);
        while (cur) |t| : (cur = c.ggml_get_next_tensor(ctx, t)) {
            const name = gguf_meta.cString(c.ggml_get_name(t));
            for (self.weights.items) |*w| {
                if (std.mem.eql(u8, w.name(), name))
                    return exc.throw(.runtime_error, "invalid model: tensor '{s}' is duplicated", .{name});
            }
            self.n_elements +%= @bitCast(c.ggml_nelements(t));
            self.n_bytes +%= c.ggml_nbytes(t);
            const w = try Weight.init(&self.files.items[self.files.items.len - 1], idx, gguf_ctx, t, exc);
            self.weights.append(allocator, w) catch oom();
        }
    }

    /// The split half of the constructor: the other files' metadata and
    /// tensors, each checked for its index.
    fn loadSplits(self: *Loader, fname: [:0]const u8, splits: *std.ArrayList([:0]const u8), n_split: u16, trace: c_int, exc: *Exception) Thrown!void {
        const kv_split_no = arch.kvName(self.meta.arch, .split_no, null);
        const idx0 = (try self.meta.getKeyNamed(u16, kv_split_no.slice(), true, exc)).?;
        if (idx0 != 0) {
            return exc.throw(.runtime_error, "illegal split file idx: {d} (file: {s}), model must be loaded with the first split", .{ idx0, fname });
        }

        if (splits.items.len == 0) try listSplits(fname, idx0, n_split, splits, exc);

        if (n_split != @as(u16, @truncate(splits.items.len))) {
            return exc.throw(.runtime_error, "invalid split count, given: {d} splits, but expected {d}", .{ splits.items.len, n_split });
        }

        if (trace > 0) impl.logInfo("%s: loading additional %d GGUFs\n", .{ "llama_model_loader", @as(c_int, n_split) });

        var zkey: [arch.max_name + 1]u8 = undefined;
        @memcpy(zkey[0..kv_split_no.len], kv_split_no.slice());
        zkey[kv_split_no.len] = 0;

        var idx: u16 = 1;
        while (idx < n_split) : (idx += 1) {
            const fname_split = splits.items[idx];
            var ctx: ?*c.ggml_context = null;
            const split_gguf = c.gguf_init_from_file(fname_split.ptr, .{ .no_alloc = true, .ctx = &ctx }) orelse
                return exc.throw(.runtime_error, "llama_model_loader: failed to load GGUF split from {s}", .{fname_split});
            defer c.gguf_free(split_gguf);

            const kid = c.gguf_find_key(split_gguf, &zkey);
            if (kid < 0) return exc.throw(.runtime_error, "missing key {s} in GGUF split {s}", .{ kv_split_no.slice(), fname_split });
            const idx_gguf = c.gguf_get_val_u16(split_gguf, kid);
            if (idx_gguf != idx) {
                return exc.throw(.runtime_error, "invalid split file idx: {d} (file: {s}), expected {d}", .{ idx_gguf, fname_split, idx });
            }

            self.files.append(allocator, try File.open(fname_split, exc)) catch oom();
            self.contexts.append(allocator, ctx.?) catch oom();
            try self.addWeights(ctx.?, idx, split_gguf, exc);
        }

        self.n_tensors = (try self.meta.getKey(c_int, .split_tensors_count, true, exc)).?;
        const n_loaded: c_int = @intCast(self.weights.items.len);
        if (self.n_tensors != n_loaded) {
            return exc.throw(.runtime_error, "corrupted model: {d} tensors expected but {d} found", .{ self.n_tensors, n_loaded });
        }

        impl.logInfo("%s: additional %d GGUFs metadata loaded.\n", .{ "llama_model_loader", @as(c_int, n_split) - 1 });
    }

    /// The file-type guess and its override, from the constructor of
    /// `llama_model_loader` (llama-model-loader.cpp:527 @c1d0e7a00).
    fn guessFtype(self: *Loader, trace: c_int, exc: *Exception) Thrown!void {
        var n_type: [c.GGML_TYPE_COUNT]u32 = @splat(0);
        var n_type_max: u32 = 0;
        var type_max: c.enum_ggml_type = c.GGML_TYPE_F32;

        for (self.weights.items) |*w| {
            const ty = w.tensor.type;
            n_type[@intCast(ty)] += 1;
            if (n_type_max < n_type[@intCast(ty)]) {
                n_type_max = n_type[@intCast(ty)];
                type_max = ty;
            }
            if (trace > 0) {
                const sid: c_int = w.idx;
                var shape: [256]u8 = undefined;
                impl.logInfo("%s: - tensor split %2d: %32s %-8s [ %s ] %8.2f MiB\n", .{
                    "llama_model_loader",                sid,                                                                                      c.ggml_get_name(w.tensor), c.ggml_type_name(ty),
                    formatTensorShape(w.tensor, &shape), @as(f64, @floatCast(@as(f32, @floatFromInt(c.ggml_nbytes(w.tensor))) / 1024.0 / 1024.0)),
                });
            }
        }

        self.ftype = switch (type_max) {
            c.GGML_TYPE_F32 => c.LLAMA_FTYPE_ALL_F32,
            c.GGML_TYPE_F16 => c.LLAMA_FTYPE_MOSTLY_F16,
            c.GGML_TYPE_BF16 => c.LLAMA_FTYPE_MOSTLY_BF16,
            c.GGML_TYPE_Q4_0 => c.LLAMA_FTYPE_MOSTLY_Q4_0,
            c.GGML_TYPE_Q4_1 => c.LLAMA_FTYPE_MOSTLY_Q4_1,
            c.GGML_TYPE_Q5_0 => c.LLAMA_FTYPE_MOSTLY_Q5_0,
            c.GGML_TYPE_Q5_1 => c.LLAMA_FTYPE_MOSTLY_Q5_1,
            c.GGML_TYPE_Q8_0 => c.LLAMA_FTYPE_MOSTLY_Q8_0,
            c.GGML_TYPE_Q2_K => c.LLAMA_FTYPE_MOSTLY_Q2_K,
            c.GGML_TYPE_Q3_K => c.LLAMA_FTYPE_MOSTLY_Q3_K_M,
            c.GGML_TYPE_Q4_K => c.LLAMA_FTYPE_MOSTLY_Q4_K_M,
            c.GGML_TYPE_Q5_K => c.LLAMA_FTYPE_MOSTLY_Q5_K_M,
            c.GGML_TYPE_Q6_K => c.LLAMA_FTYPE_MOSTLY_Q6_K,
            c.GGML_TYPE_TQ1_0 => c.LLAMA_FTYPE_MOSTLY_TQ1_0,
            c.GGML_TYPE_TQ2_0 => c.LLAMA_FTYPE_MOSTLY_TQ2_0,
            c.GGML_TYPE_IQ2_XXS => c.LLAMA_FTYPE_MOSTLY_IQ2_XXS,
            c.GGML_TYPE_IQ2_XS => c.LLAMA_FTYPE_MOSTLY_IQ2_XS,
            c.GGML_TYPE_IQ2_S => c.LLAMA_FTYPE_MOSTLY_IQ2_S,
            c.GGML_TYPE_IQ3_XXS => c.LLAMA_FTYPE_MOSTLY_IQ3_XXS,
            c.GGML_TYPE_IQ1_S => c.LLAMA_FTYPE_MOSTLY_IQ1_S,
            c.GGML_TYPE_IQ1_M => c.LLAMA_FTYPE_MOSTLY_IQ1_M,
            c.GGML_TYPE_IQ4_NL => c.LLAMA_FTYPE_MOSTLY_IQ4_NL,
            c.GGML_TYPE_IQ4_XS => c.LLAMA_FTYPE_MOSTLY_IQ4_XS,
            c.GGML_TYPE_IQ3_S => c.LLAMA_FTYPE_MOSTLY_IQ3_S,
            c.GGML_TYPE_NVFP4 => c.LLAMA_FTYPE_MOSTLY_NVFP4,
            c.GGML_TYPE_Q1_0 => c.LLAMA_FTYPE_MOSTLY_Q1_0,
            c.GGML_TYPE_Q2_0 => c.LLAMA_FTYPE_MOSTLY_Q2_0,
            else => blk: {
                impl.logWarn("%s: unknown type %s\n", .{ "llama_model_loader", c.ggml_type_name(type_max) });
                break :blk c.LLAMA_FTYPE_ALL_F32;
            },
        };

        // This is a way to mark that the file type was guessed.
        self.ftype |= c.LLAMA_FTYPE_GUESSED;

        if (try self.meta.getKey(u32, .general_file_type, false, exc)) |v| self.ftype = v;

        self.n_type_counts = n_type;
    }

    /// The metadata dump and the type counts, from the constructor of
    /// `llama_model_loader` (llama-model-loader.cpp:527 @c1d0e7a00).
    fn dumpMetadata(self: *Loader) void {
        const ctx = self.meta.ctx;
        impl.logInfo("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", .{"llama_model_loader"});

        var i: c_int = 0;
        while (i < self.n_kv) : (i += 1) {
            const name = c.gguf_get_key(ctx, i);
            const ty = c.gguf_get_kv_type(ctx, i);
            var type_name: [128]u8 = undefined;
            const tn: [*:0]const u8 = if (ty == c.GGUF_TYPE_ARRAY) blk: {
                _ = c.snprintf(&type_name, type_name.len, "%s[%s,%zu]", c.gguf_type_name(ty), c.gguf_type_name(c.gguf_get_arr_type(ctx, i)), c.gguf_get_arr_n(ctx, i));
                break :blk @ptrCast(&type_name);
            } else c.gguf_type_name(ty);

            var value: std.ArrayList(u8) = .empty;
            defer value.deinit(allocator);
            kvToStr(ctx, i, &value);
            const max_value_len = 40;
            if (value.items.len > max_value_len) {
                value.shrinkRetainingCapacity(max_value_len - 3);
                value.appendSlice(allocator, "...") catch oom();
            }
            replaceAll(&value, "\n", "\\n");

            impl.logInfo("%s: - kv %3d: %42s %-16s = %.*s\n", .{
                "llama_model_loader", i, name, tn, @as(c_int, @intCast(value.items.len)), value.items.ptr,
            });
        }

        // `std::map<ggml_type, uint32_t>`: in type order, the zeros skipped.
        for (self.n_type_counts, 0..) |n, ty| {
            if (n == 0) continue;
            impl.logInfo("%s: - type %4s: %4d tensors\n", .{ "llama_model_loader", c.ggml_type_name(@intCast(ty)), @as(c_int, @bitCast(n)) });
        }
    }

    /// Return: nothing.
    pub fn deinit(self: *Loader) void {
        for (self.files.items) |*f| f.close();
        self.files.deinit(allocator);
        for (self.contexts.items) |ctx| c.ggml_free(ctx);
        self.contexts.deinit(allocator);
        self.weights.deinit(allocator);
        if (self.owned) |m| c.gguf_free(m);
        self.meta.deinit();
        self.arch_name.deinit(allocator);
    }

    /// Ports `llama_model_loader`'s `get_arch_name` (llama-model-loader.cpp:827
    /// @c1d0e7a00).
    pub fn getArchName(self: *const Loader) []const u8 {
        return self.arch_name.items;
    }

    /// Ports `llama_model_loader`'s `get_arch` (llama-model-loader.cpp:831
    /// @c1d0e7a00).
    pub fn getArch(self: *const Loader) arch.Arch {
        return self.meta.arch;
    }

    /// Ports `llama_model_loader`'s `ftype_name` (llama-model-loader.cpp:1710
    /// @c1d0e7a00).
    pub fn ftypeNameOf(self: *const Loader) [:0]const u8 {
        return ftypeName(self.ftype);
    }

    /// Ports `llama_model_loader`'s `print_info` (llama-model-loader.cpp:1714
    /// @c1d0e7a00).
    pub fn printInfo(self: *const Loader) void {
        impl.logInfo("%s: file format = %s\n", .{ "print_info", fileVersionName(self.fver).ptr });
        impl.logInfo("%s: file type   = %s\n", .{ "print_info", ftypeName(self.ftype).ptr });
        const bytes: f64 = @floatFromInt(self.n_bytes);
        const bpw = bytes * 8.0 / @as(f64, @floatFromInt(self.n_elements));
        if (self.n_bytes < GiB) {
            impl.logInfo("%s: file size   = %.2f MiB (%.2f BPW) \n", .{ "print_info", bytes / 1024.0 / 1024.0, bpw });
        } else {
            impl.logInfo("%s: file size   = %.2f GiB (%.2f BPW) \n", .{ "print_info", bytes / 1024.0 / 1024.0 / 1024.0, bpw });
        }
    }
};

/// Ports `llama_get_list_splits` (llama-model-loader.cpp:82 @c1d0e7a00),
/// through the public `llama_split_prefix` and `llama_split_path`.
fn listSplits(path: [:0]const u8, idx: u16, n_split: u16, out: *std.ArrayList([:0]const u8), exc: *Exception) Thrown!void {
    var buf: [std.c.PATH_MAX]u8 = std.mem.zeroes([std.c.PATH_MAX]u8);
    const ret = c.llama_split_prefix(&buf, buf.len, path.ptr, idx, n_split);
    if (ret == 0) return exc.throw(.runtime_error, "invalid split file name: {s}", .{path});
    const prefix = allocator.dupeSentinel(u8, buf[0..@intCast(ret)], 0) catch oom();
    defer allocator.free(prefix);
    if (prefix.len == 0) return exc.throw(.runtime_error, "invalid split file: {s}", .{path});

    var i: c_int = 0;
    while (i < n_split) : (i += 1) {
        const n = c.llama_split_path(&buf, buf.len, prefix.ptr, i, n_split);
        out.append(allocator, allocator.dupeSentinel(u8, buf[0..@intCast(n)], 0) catch oom()) catch oom();
    }
}

/// Ports `llama_format_tensor_shape` for a tensor (llama-impl.cpp:110 @c1d0e7a00).
fn formatTensorShape(t: *const c.ggml_tensor, buf: *[256]u8) [*:0]const u8 {
    var len: usize = @intCast(c.snprintf(buf, buf.len, "%6lld", @as(c_longlong, t.ne[0])));
    var i: usize = 1;
    while (i < c.GGML_MAX_DIMS) : (i += 1) {
        len += @intCast(c.snprintf(buf[len..].ptr, buf.len - len, ", %6lld", @as(c_longlong, t.ne[i])));
    }
    return @ptrCast(buf);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a guessed file type carries the prefix and a stored one does not" {
    try std.testing.expectEqualStrings("(guessed) Q4_K - Medium", ftypeName(c.LLAMA_FTYPE_MOSTLY_Q4_K_M | c.LLAMA_FTYPE_GUESSED));
    try std.testing.expectEqualStrings("Q4_K - Medium", ftypeName(c.LLAMA_FTYPE_MOSTLY_Q4_K_M));
    try std.testing.expectEqualStrings("unknown, may not work", ftypeName(999));
}

test "replace_all leaves an empty search alone and replaces every match" {
    var s: std.ArrayList(u8) = .empty;
    defer s.deinit(allocator);
    try s.appendSlice(allocator, "a\"b\"\\");
    replaceAll(&s, "", "x");
    try std.testing.expectEqualStrings("a\"b\"\\", s.items);
    replaceAll(&s, "\\", "\\\\");
    replaceAll(&s, "\"", "\\\"");
    try std.testing.expectEqualStrings("a\\\"b\\\"\\\\", s.items);
}

test "the block index is sscanf's" {
    try std.testing.expectEqual(@as(c_int, 12), blockOf("blk.12.attn_q.weight"));
    try std.testing.expectEqual(@as(c_int, -1), blockOf("output.weight"));
    try std.testing.expectEqual(@as(c_int, -3), blockOf("blk.-3.x"));
}
