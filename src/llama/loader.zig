//! Opening a model: its metadata, its file type, and its tensors.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/llama-model-loader.cpp` -- `llama_file_version_name`,
//!   `llama_ftype_name`, `llama_get_list_splits`, `weight_buft_supported`,
//!   `select_weight_buft`, and every member of `llama_model_loader`
//! - `llama.cpp/src/llama-model-loader.h` -- `llama_fver`,
//!   `llama_tensor_weight`, `weight_name_comparer`, the `TENSOR_*` flags and
//!   `buft_list_t`
//! - `llama.cpp/src/llama-arch.h` -- `LLM_TN_IMPL`
//! - `llama.cpp/src/llama-impl.cpp` -- `replace_all`, `gguf_data_to_str`
//!   and `gguf_kv_to_str`
//!
//! All at v0.3.0 (`c1d0e7a00`). The typed getters the constructor and the
//! model use are `gguf_meta.zig`'s.
//!
//! # Scope
//!
//! All of `llama_model_loader`: the metadata half, everything up to and
//! including `print_info`, which is what loading hyperparameters needs; and
//! the tensor half -- creating each weight's metadata in the context of the
//! buffer type chosen for it, mapping the files, and loading the data.
//! `llama_file` and the mappings are `mmap.zig`'s.
//!
//! # The weights' order is the C++ map's
//!
//! `weights_map` is a `std`'s `map` under `weight_name_comparer`: by the
//! block index `sscanf` reads from `blk.%d.`, then by name. The file type
//! is guessed from it in that order, and the first type to reach the
//! highest count wins a tie, so the order is part of the answer. `Weight`s
//! are sorted the same way here, and the block index is read by libc's own
//! `sscanf` so its edge cases are the C++'s.

const std = @import("std");
const impl = @import("impl.zig");
const arch = @import("arch.zig");
const gguf_meta = @import("gguf_meta.zig");
const hparams_mod = @import("hparams.zig");
const mmap = @import("mmap.zig");
const regex = @import("regex.zig");

const File = mmap.File;
const Hparams = hparams_mod.Hparams;

const c = impl.c;
const allocator = impl.allocator;
const Exception = impl.Exception;
const Thrown = impl.Thrown;

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
/// as `std`'s `to_string` writes it, so a float has six decimals.
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

/// Ports `llama_model_loader` (llama-model-loader.h:31 @c1d0e7a00).
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

    // The tensor half.

    /// `weights_map`'s lookup: a weight's name to its index in `weights`.
    weight_index: std.StringHashMapUnmanaged(usize) = .empty,
    /// The caller's buffer-type overrides, null-pattern terminated, or null.
    tensor_buft_overrides: ?[*]const c.llama_model_tensor_buft_override = null,
    /// How a model built from metadata alone gets its tensors' data.
    set_tensor_data: c.llama_model_set_tensor_data_t = null,
    set_tensor_data_ud: ?*anyopaque = null,

    /// `mappings`: one per file once `initMappings` has run with mmap on.
    mappings: std.ArrayList(mmap.Mmap) = .empty,
    size_done: usize = 0,
    size_data: usize = 0,
    /// `mmaps_used`: the byte range of each mapping that tensors use.
    mmaps_used: std.ArrayList([2]usize) = .empty,

    /// `ctx_map`: one tensor context per buffer type, kept in the order of
    /// the buffer types' names, as the C++'s comparator orders them.
    ctx_map: std.ArrayList(CtxEntry) = .empty,

    n_tensors_moved: usize = 0,
    first_tensor_moved_name: std.ArrayList(u8) = .empty,
    first_tensor_moved_type_name: [*:0]const u8 = "",
    first_moved_from_buft: c.ggml_backend_buffer_type_t = null,
    first_moved_to_buft: c.ggml_backend_buffer_type_t = null,

    /// An entry of `ctx_map`. The model takes `ctx` when it allocates the
    /// buffers, and nulls it here.
    pub const CtxEntry = struct {
        buft: c.ggml_backend_buffer_type_t,
        ctx: ?*c.ggml_context,
    };

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

                self.files.append(allocator, try File.open(path, "rb", self.use_direct_io, exc)) catch oom();
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
        for (self.weights.items, 0..) |*w, i| self.weight_index.put(allocator, w.name(), i) catch oom();

        self.n_kv = @truncate(c.gguf_get_n_kv(self.meta.ctx));
        self.n_tensors = @intCast(self.weights.items.len);
        self.fver = @fromBackingInt(@intCast(c.gguf_get_version(self.meta.ctx)));

        const shown: []const u8 = if (fname.len == 0) "(file*)" else fname;
        impl.logInfo("%s: loaded meta data with %d key-value pairs and %d tensors from %.*s (version %s)\n", .{
            "llama_model_loader", self.n_kv, self.n_tensors, @as(c_int, @intCast(shown.len)), shown.ptr, fileVersionName(self.fver).ptr,
        });

        try self.guessFtype(trace, exc);
        self.dumpMetadata();

        // `llama_mmap`'s `SUPPORTED` holds on every target this builds for.
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

            self.files.append(allocator, try File.open(fname_split, "rb", self.use_direct_io, exc)) catch oom();
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
        for (self.mappings.items) |*m| m.deinit();
        self.mappings.deinit(allocator);
        self.mmaps_used.deinit(allocator);
        for (self.ctx_map.items) |e| if (e.ctx) |ctx| c.ggml_free(ctx);
        self.ctx_map.deinit(allocator);
        self.first_tensor_moved_name.deinit(allocator);
        self.weight_index.deinit(allocator);
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

    // -------------------------------------------------------------------------
    // The tensor half

    /// Ports `llama_model_loader`'s `get_weight` (llama-model-loader.cpp:835
    /// @c1d0e7a00).
    ///
    /// Return: the weight, borrowed from the loader, or null.
    pub fn getWeight(self: *const Loader, name: []const u8) ?*Weight {
        const i = self.weight_index.get(name) orelse return null;
        return &self.weights.items[i];
    }

    /// Ports `llama_model_loader`'s `require_weight` (llama-model-loader.cpp:844
    /// @c1d0e7a00).
    pub fn requireWeight(self: *const Loader, name: []const u8, exc: *Exception) Thrown!*Weight {
        return self.getWeight(name) orelse exc.throw(.runtime_error, "require_weight: tensor '{s}' not found", .{name});
    }

    /// Ports `llama_model_loader`'s `get_tensor_meta` (llama-model-loader.cpp:852
    /// @c1d0e7a00).
    pub fn getTensorMeta(self: *const Loader, name: []const u8) ?*c.ggml_tensor {
        const w = self.getWeight(name) orelse return null;
        return w.tensor;
    }

    /// Ports `llama_model_loader`'s `check_tensor_dims` (llama-model-loader.cpp:868
    /// @c1d0e7a00).
    ///
    /// Return: the tensor's metadata; null when it is absent and not
    /// required; `error.Thrown` when it is absent and required, or its
    /// shape is not `ne`.
    pub fn checkTensorDims(self: *const Loader, name: []const u8, ne: []const i64, required: bool, allow_reshape: bool, exc: *Exception) Thrown!?*c.ggml_tensor {
        const cur = self.getTensorMeta(name) orelse {
            if (!required) return null;
            return exc.throw(.runtime_error, "check_tensor_dims: tensor '{s}' not found", .{name});
        };

        var is_ok = true;
        if (allow_reshape) {
            const ncur = c.ggml_nelements(cur);
            var nexp: i64 = 1;
            for (ne) |n| nexp *%= n;
            if (ncur != nexp) is_ok = false;
        } else {
            for (0..c.GGML_MAX_DIMS) |i| {
                if ((i < ne.len and ne[i] != cur.ne[i]) or (i >= ne.len and cur.ne[i] != 1)) {
                    is_ok = false;
                    break;
                }
            }
        }

        if (!is_ok) {
            var want: [256]u8 = undefined;
            var got: [256]u8 = undefined;
            return exc.throw(.runtime_error, "check_tensor_dims: tensor '{s}' has wrong shape; expected {s}, got {s}", .{
                name, formatShape(ne, &want), std.mem.span(formatTensorShape(cur, &got)),
            });
        }
        return cur;
    }

    /// Ports `llama_model_loader`'s `create_tensor` (llama-model-loader.cpp:1068
    /// @c1d0e7a00): the tensor's metadata in the context of the buffer type
    /// it will live in, with that buffer type chosen here.
    ///
    /// Parameters:
    /// - `hparams`: the model's hyperparameters, for the op probes.
    /// - `buft_list_cpu`, `buft_list_input`, `buft_list_output`,
    ///   `buft_list_layer`: where each kind of tensor may live; the last is
    ///   null for a tensor with no block.
    /// - `tn`: the tensor's name.
    /// - `ne`: the expected shape.
    /// - `flags`: `tensor_not_required` and the rest.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: the tensor, owned by its context; null when it is optional
    /// and absent, or skipped; `error.Thrown` where the C++ throws.
    pub fn createTensor(
        self: *Loader,
        hparams: *const Hparams,
        buft_list_cpu: *const BuftList,
        buft_list_input: *const BuftList,
        buft_list_output: *const BuftList,
        buft_list_layer: ?*const BuftList,
        tn: Tn,
        ne: []const i64,
        flags: u32,
        exc: *Exception,
    ) Thrown!?*c.ggml_tensor {
        const name = tn.str();
        if (self.files.items.len == 0) {
            if (flags & tensor_skip_if_virtual != 0) return null;
            var ty: c.enum_ggml_type = c.GGML_TYPE_F32;
            var zname: [arch.max_name + 1]u8 = undefined;
            const nm = zeroTerminate(&zname, name.slice());
            const tid = c.gguf_find_tensor(self.meta.ctx, nm);
            if (tid != -1) ty = c.gguf_get_tensor_type(self.meta.ctx, tid);

            // For tensors that are not required some of the dimensions can be invalid.
            if (flags & tensor_not_required != 0) {
                for (ne) |n| if (n <= 0) return null;
            }

            var t_meta: c.ggml_tensor = std.mem.zeroes(c.ggml_tensor);
            t_meta.type = ty;
            for (0..c.GGML_MAX_DIMS) |dim| {
                t_meta.ne[dim] = if (dim < ne.len) ne[dim] else 1;
                impl.assert(t_meta.ne[dim] >= 1, "llama-model-loader.cpp", 1259, "t_meta.ne[dim] >= 1");
                if (dim == 0) {
                    t_meta.nb[dim] = c.ggml_type_size(ty);
                } else if (dim == 1) {
                    t_meta.nb[dim] = c.ggml_row_size(ty, t_meta.ne[dim - 1]);
                } else {
                    t_meta.nb[dim] = t_meta.nb[dim - 1] *% @as(usize, @bitCast(t_meta.ne[dim - 1]));
                }
                impl.assert(t_meta.nb[dim] >= 1, "llama-model-loader.cpp", 1267, "t_meta.nb[dim] >= 1");
            }
            _ = c.ggml_set_name(&t_meta, nm);

            const buft = (try self.buftForTensor(hparams, buft_list_cpu, buft_list_input, buft_list_output, buft_list_layer, tn, &t_meta, flags, exc)).?;
            const ctx = try self.ctxForBuft(hparams, buft, exc);
            const ret = c.ggml_dup_tensor(ctx, &t_meta);
            _ = c.ggml_set_name(ret, nm);
            return ret;
        }

        impl.logDebug("%s: loading tensor %.*s\n", .{ "create_tensor", @as(c_int, @intCast(name.len)), &name.buf });
        const cur = (try self.checkTensorDims(name.slice(), ne, flags & tensor_not_required == 0, flags & tensor_allow_reshape != 0, exc)) orelse return null;

        var t_meta: c.ggml_tensor = cur.*;
        if (flags & tensor_allow_reshape != 0) {
            for (0..c.GGML_MAX_DIMS) |dim| {
                t_meta.ne[dim] = if (dim < ne.len) ne[dim] else 1;
                if (dim == 0) {
                    t_meta.nb[dim] = c.ggml_type_size(t_meta.type);
                } else if (dim == 1) {
                    t_meta.nb[dim] = c.ggml_row_size(t_meta.type, t_meta.ne[dim - 1]);
                } else {
                    t_meta.nb[dim] = @as(usize, @bitCast(t_meta.ne[dim - 1])) *% t_meta.nb[dim - 1];
                }
            }
        }

        impl.assert(c.ggml_nbytes(&t_meta) == c.ggml_nbytes(cur), "llama-model-loader.cpp", 1299, "ggml_nbytes(&t_meta) == ggml_nbytes(cur)");

        const buft = (try self.buftForTensor(hparams, buft_list_cpu, buft_list_input, buft_list_output, buft_list_layer, tn, &t_meta, flags, exc)) orelse return null;
        const ctx = try self.ctxForBuft(hparams, buft, exc);

        // If duplicated, check if the original tensor was allocated in the
        // same buffer type context and avoid creating a new one.
        var zname: [arch.max_name + 1]u8 = undefined;
        const nm = zeroTerminate(&zname, name.slice());
        if (flags & tensor_duplicated != 0) {
            if (c.ggml_get_tensor(ctx, nm)) |t| return t;
        }

        const duplicated = flags & tensor_duplicated != 0;
        const tensor = c.ggml_dup_tensor(ctx, &t_meta);
        _ = c.ggml_set_name(tensor, c.ggml_get_name(&t_meta));

        if (duplicated) {
            self.size_data +%= c.ggml_nbytes(&t_meta);
        } else {
            self.n_created += 1;
        }
        return tensor;
    }

    /// `ctx_for_buft`, the first lambda of `create_tensor`
    /// (llama-model-loader.cpp:1068 @c1d0e7a00): one context per buffer
    /// type, created on first use.
    fn ctxForBuft(self: *Loader, hparams: *const Hparams, buft: c.ggml_backend_buffer_type_t, exc: *Exception) Thrown!*c.ggml_context {
        const name = std.mem.span(c.ggml_backend_buft_name(buft));
        var pos: usize = 0;
        while (pos < self.ctx_map.items.len) : (pos += 1) {
            const order = std.mem.order(u8, std.mem.span(c.ggml_backend_buft_name(self.ctx_map.items[pos].buft)), name);
            if (order == .eq) return self.ctx_map.items[pos].ctx.?;
            if (order == .gt) break;
        }

        var max_n_tensors: c_int = self.n_tensors;
        max_n_tensors += 1; // duplicated output tensor
        max_n_tensors +%= @as(c_int, @bitCast(hparams.nLayer() *% 2)); // duplicated rope freq tensors
        if (self.files.items.len == 0) {
            max_n_tensors +%= @as(c_int, @bitCast(hparams.nLayer() *% 256));
        }
        const ctx_size = c.ggml_tensor_overhead() *% @as(usize, @intCast(max_n_tensors));
        const ctx = c.ggml_init(.{ .mem_size = ctx_size, .mem_buffer = null, .no_alloc = true }) orelse
            return exc.throw(.runtime_error, "failed to create ggml context", .{});
        self.ctx_map.insert(allocator, pos, .{ .buft = buft, .ctx = ctx }) catch oom();
        return ctx;
    }

    /// `buft_for_tensor`, the second lambda of `create_tensor`
    /// (llama-model-loader.cpp:1068 @c1d0e7a00): which buffer type a
    /// tensor goes in.
    fn buftForTensor(
        self: *Loader,
        hparams: *const Hparams,
        buft_list_cpu: *const BuftList,
        buft_list_input: *const BuftList,
        buft_list_output: *const BuftList,
        buft_list_layer: ?*const BuftList,
        tn: Tn,
        t_meta_: ?*c.ggml_tensor,
        flags: u32,
        exc: *Exception,
    ) Thrown!c.ggml_backend_buffer_type_t {
        const name = tn.str();
        const t_meta = t_meta_ orelse {
            if (flags & tensor_not_required != 0) return null;
            return exc.throw(.runtime_error, "missing tensor '{s}'", .{name.slice()});
        };

        // Some models use the token embedding tensor as the output; loaded
        // as the output, it is a duplicate.
        var tn_tensor = tn.tensor;
        if (tn.tensor == .token_embd and flags & tensor_duplicated != 0) tn_tensor = .output;

        var inner: Exception = .{};
        const info = arch.tensorInfoFor(tn_tensor, &inner) catch
            return exc.throw(.runtime_error, "missing tensor info mapping for {s}", .{name.slice()});

        // Skip unused tensors.
        if (info.op == c.GGML_OP_NONE or flags & tensor_skip != 0) {
            const nbytes = c.ggml_nbytes(t_meta);
            impl.logWarn("model has unused tensor %.*s (size = %zu bytes) -- ignoring\n", .{ @as(c_int, @intCast(name.len)), &name.buf, nbytes });
            self.size_data -%= nbytes;
            self.n_created += 1;
            return null;
        }

        // Tensors with a "bias" suffix are always used with ADD or ADD_ID;
        // embedded-adapter ".lora_a"/".lora_b" tensors with MUL_MAT_ID.
        var op: c.enum_ggml_op = undefined;
        if (tn.suffix != null and std.mem.eql(u8, tn.suffix.?, "bias")) {
            op = if (info.op == c.GGML_OP_MUL_MAT_ID) c.GGML_OP_ADD_ID else c.GGML_OP_ADD;
        } else if (hparams.router_layer >= 0 and tn.suffix != null and
            (std.mem.eql(u8, tn.suffix.?, "lora_a") or std.mem.eql(u8, tn.suffix.?, "lora_b")))
        {
            op = c.GGML_OP_MUL_MAT_ID;
        } else {
            op = info.op;
        }

        // Sanity checks.
        if (info.layer == .input or info.layer == .output) {
            if (tn.bid != -1) c.ggml_abort("llama-model-loader.cpp", 1150, "input/output layer tensor %.*s used with a layer number", @as(c_int, @intCast(name.len)), &name.buf);
        } else {
            if (tn.bid == -1) c.ggml_abort("llama-model-loader.cpp", 1154, "repeating layer tensor %.*s used without a layer number", @as(c_int, @intCast(name.len)), &name.buf);
        }

        // Select the buffer type for this tensor.
        const buft_list: *const BuftList = switch (info.layer) {
            .input => buft_list_input,
            .output => buft_list_output,
            .repeating => blk: {
                impl.assert(buft_list_layer != null, "llama-model-loader.cpp", 1168, "buft_list_layer != nullptr");
                break :blk buft_list_layer.?;
            },
        };

        var buft: c.ggml_backend_buffer_type_t = null;

        // Check overrides.
        if (self.tensor_buft_overrides) |overrides_| {
            var overrides = overrides_;
            while (overrides[0].pattern != null) : (overrides += 1) {
                var re = regex.Regex.compile(std.mem.span(overrides[0].pattern)) catch |e|
                    return exc.throw(.runtime_error, "{s}", .{regex.errorText(e)});
                defer re.deinit();
                var m = regex.Match.init();
                defer m.deinit();
                if (re.search(name.slice(), &m)) {
                    if (overrides[0].buft == c.ggml_backend_cpu_buffer_type()) {
                        // When overriding to a CPU buffer, consider the extra buffer types.
                        buft = try selectWeightBuft(hparams, t_meta, op, buft_list_cpu, exc);
                        if (self.use_mmap and !warned_cpu_override) {
                            warned_cpu_override = true;
                            impl.logWarn("llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none for better performance\n", .{});
                        }
                    } else {
                        buft = overrides[0].buft;
                    }
                    impl.logDebug("tensor %.*s (%zu MiB %s) buffer type overridden to %s\n", .{
                        @as(c_int, @intCast(name.len)),      &name.buf,
                        c.ggml_nbytes(t_meta) / 1024 / 1024, c.ggml_type_name(t_meta.type),
                        c.ggml_backend_buft_name(buft),
                    });
                    break;
                }
            }
        }

        if (buft == null) {
            buft = try selectWeightBuft(hparams, t_meta, op, buft_list, exc);
            if (buft == null) return exc.throw(.runtime_error, "failed to find a compatible buffer type for tensor {s}", .{name.slice()});
        }

        // Avoid using a host buffer when using mmap.
        const buft_dev = c.ggml_backend_buft_get_device(buft);
        if (self.use_mmap and buft_dev != null and buft == c.ggml_backend_dev_host_buffer_type(buft_dev)) {
            const cpu_dev = c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU) orelse
                return exc.throw(.runtime_error, "no CPU backend found", .{});
            buft = c.ggml_backend_dev_buffer_type(cpu_dev);
        }

        if (buft != buft_list.items[0].buft) {
            if (self.n_tensors_moved == 0) {
                self.first_tensor_moved_name.clearRetainingCapacity();
                self.first_tensor_moved_name.appendSlice(allocator, gguf_meta.cString(&t_meta.name)) catch oom();
                self.first_tensor_moved_type_name = c.ggml_type_name(t_meta.type);
                self.first_moved_from_buft = buft_list.items[0].buft;
                self.first_moved_to_buft = buft;
            }
            self.n_tensors_moved += 1;
        }
        return buft;
    }

    /// Ports `llama_model_loader`'s `done_getting_tensors`
    /// (llama-model-loader.cpp:1330 @c1d0e7a00).
    ///
    /// Return: nothing, or `error.Thrown` when the count is wrong.
    pub fn doneGettingTensors(self: *const Loader, partial: bool, exc: *Exception) Thrown!void {
        if (self.n_created > self.n_tensors) {
            return exc.throw(.runtime_error, "done_getting_tensors: too many tensors created; expected {d}, got {d}", .{ self.n_tensors, self.n_created });
        }
        if (self.n_created < self.n_tensors) {
            if (!partial) {
                return exc.throw(.runtime_error, "done_getting_tensors: wrong number of tensors; expected {d}, got {d}", .{ self.n_tensors, self.n_created });
            }
            impl.logInfo("%s: partial load \xe2\x80\x94 used %d of %d tensors in the file (rest belong to a sibling model on the same .gguf)\n", .{
                "done_getting_tensors", self.n_created, self.n_tensors,
            });
        }
        if (self.n_tensors_moved > 0) {
            impl.logDebug("%s: tensor '%.*s' (%s) (and %zu others) cannot be used with preferred buffer type %s, using %s instead\n", .{
                "done_getting_tensors",
                @as(c_int, @intCast(self.first_tensor_moved_name.items.len)),
                self.first_tensor_moved_name.items.ptr,
                self.first_tensor_moved_type_name,
                self.n_tensors_moved - 1,
                c.ggml_backend_buft_name(self.first_moved_from_buft),
                c.ggml_backend_buft_name(self.first_moved_to_buft),
            });
        }
    }

    /// Ports `llama_model_loader`'s `init_mappings` (llama-model-loader.cpp:1348
    /// @c1d0e7a00).
    ///
    /// Parameters:
    /// - `prefetch`: advise the kernel the whole file will be needed.
    /// - `mlock_mmaps`: receives one lock per mapping, or null for none.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: nothing, or `error.Thrown` when a file cannot be mapped.
    pub fn initMappings(self: *Loader, prefetch: bool, mlock_mmaps: ?*std.ArrayList(*mmap.Mlock), exc: *Exception) Thrown!void {
        if (self.use_mmap) {
            self.mappings.ensureTotalCapacity(allocator, self.files.items.len) catch oom();
            self.mmaps_used.ensureTotalCapacity(allocator, self.files.items.len) catch oom();
            for (self.files.items) |*file| {
                var is_numa = false;
                if (c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU)) |dev| {
                    const reg = c.ggml_backend_dev_backend_reg(dev);
                    const is_numa_fn: ?*const fn () callconv(.c) bool = @ptrCast(@alignCast(c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_is_numa")));
                    if (is_numa_fn) |f| is_numa = f();
                }

                const mapping = try mmap.Mmap.init(file, if (prefetch) std.math.maxInt(usize) else 0, is_numa, exc);
                self.mmaps_used.append(allocator, .{ mapping.size, 0 }) catch oom();
                if (mlock_mmaps) |locks| {
                    const lock = allocator.create(mmap.Mlock) catch oom();
                    lock.* = .{};
                    lock.init(mapping.addr);
                    locks.append(allocator, lock) catch oom();
                }
                self.mappings.append(allocator, mapping) catch oom();
            }
        }

        // Compute the total size of all tensors for progress reporting.
        for (self.weights.items) |*w| self.size_data +%= c.ggml_nbytes(w.tensor);
    }

    /// Ports `llama_model_loader`'s `get_mapping_range` (llama-model-loader.cpp:1381
    /// @c1d0e7a00): the byte range of mapping `idx` that `ctx`'s tensors use.
    pub fn getMappingRange(self: *const Loader, first: *usize, last: *usize, addr: **anyopaque, idx: usize, ctx: *c.ggml_context) void {
        impl.assert(self.mappings.items.len != 0, "llama-model-loader.cpp", 1382, "!mappings.empty()");
        const mapping = &self.mappings.items[idx];
        first.* = mapping.size;
        last.* = 0;
        addr.* = mapping.addr;
        var t: ?*c.ggml_tensor = c.ggml_get_first_tensor(ctx);
        while (t) |tensor| : (t = c.ggml_get_next_tensor(ctx, tensor)) {
            const w = self.getWeight(gguf_meta.cString(c.ggml_get_name(tensor))) orelse continue;
            if (w.idx != idx) continue;
            first.* = @min(first.*, w.offs);
            last.* = @max(last.*, w.offs +% c.ggml_nbytes(tensor));
        }
    }

    /// Ports `llama_model_loader`'s `unmap_weight` (llama-model-loader.cpp:1398
    /// @c1d0e7a00).
    pub fn unmapWeight(self: *Loader, w: *const Weight) void {
        if (!self.use_mmap) return;
        self.mappings.items[w.idx].unmapFragment(w.offs, w.offs +% c.ggml_nbytes(w.tensor));
    }

    /// Ports `llama_model_loader`'s `load_data_for` (llama-model-loader.cpp:1403
    /// @c1d0e7a00), which does not go through a backend.
    pub fn loadDataFor(self: *const Loader, cur: *c.ggml_tensor, exc: *Exception) Thrown!void {
        const w = try self.requireWeight(gguf_meta.cString(c.ggml_get_name(cur)), exc);
        if (self.use_mmap) {
            const mapping = &self.mappings.items[w.idx];
            const src: [*]u8 = @ptrFromInt(@intFromPtr(mapping.addr) + w.offs);
            if (cur.data == null) {
                cur.data = src;
            } else {
                @memcpy(@as([*]u8, @ptrCast(cur.data.?))[0..c.ggml_nbytes(cur)], src[0..c.ggml_nbytes(cur)]);
            }
        } else {
            impl.assert(cur.data != null, "llama-model-loader.cpp", 1414, "cur->data != nullptr");
            impl.assert(w.idx < self.files.items.len, "llama-model-loader.cpp", 1415, "w.idx < files.size()");
            const file = &self.files.items[w.idx];
            try file.seek(w.offs, c.SEEK_SET, exc);
            try file.readRaw(cur.data.?, c.ggml_nbytes(cur), exc);
        }
        if (self.check_tensors and !c.ggml_validate_row_data(cur.type, cur.data, c.ggml_nbytes(cur))) {
            return exc.throw(.runtime_error, "tensor '{s}' has invalid data", .{gguf_meta.cString(c.ggml_get_name(cur))});
        }
    }

    /// Ports `llama_model_loader`'s `load_all_data` (llama-model-loader.cpp:1426
    /// @c1d0e7a00): every tensor of `ctx` gets its data, from the mapping or
    /// the file.
    ///
    /// The C++ validates tensors on `std`'s `async` threads and reports them in
    /// launch order after loading; here they are validated in that order
    /// after loading, which logs and throws exactly the same.
    ///
    /// Parameters:
    /// - `ctx`: the tensors to load.
    /// - `bufs`: the buffer for each file index.
    /// - `lmlocks`: one lock per file to grow as mapped tensors are used, or
    ///   null.
    /// - `progress_callback`, `progress_callback_user_data`: progress, or
    ///   null.
    /// - `exc`: receives what is thrown.
    ///
    /// Return: false when the progress callback cancelled, or `error.Thrown`.
    pub fn loadAllData(
        self: *Loader,
        ctx: *c.ggml_context,
        bufs: *const BufMap,
        lmlocks: ?*std.ArrayList(*mmap.Mlock),
        progress_callback: c.llama_progress_callback,
        progress_callback_user_data: ?*anyopaque,
        exc: *Exception,
    ) Thrown!bool {
        if (self.files.items.len == 0) {
            var t: ?*c.ggml_tensor = c.ggml_get_first_tensor(ctx);
            while (t) |tensor| : (t = c.ggml_get_next_tensor(ctx, tensor)) self.set_tensor_data.?(tensor, self.set_tensor_data_ud);
            return true;
        }
        impl.assert(self.size_data != 0, "llama-model-loader.cpp", 1438, "size_data != 0 && \"call init_mappings() first\"");

        var read_buf: std.ArrayList(u8) = .empty;
        defer read_buf.deinit(allocator);
        var validation: std.ArrayList(Validation) = .empty;
        defer validation.deinit(allocator);

        // 4 staging buffers for async uploads, each sized 1MB.
        const n_buffers = 4;

        var alignment: usize = 1;
        for (self.files.items) |*file| alignment = @max(file.readAlignment(), alignment);

        const buffer_size: usize = if (alignment != 1) 64 * 1024 * 1024 + 2 * alignment else 1 * 1024 * 1024;

        var host_buffers: std.ArrayList(c.ggml_backend_buffer_t) = .empty;
        defer host_buffers.deinit(allocator);
        var events: std.ArrayList(c.ggml_backend_event_t) = .empty;
        defer events.deinit(allocator);
        var host_ptrs: std.ArrayList(?*anyopaque) = .empty;
        defer host_ptrs.deinit(allocator);
        var buffer_idx: usize = 0;
        const upload_backend = self.uploadBackend(bufs, n_buffers, buffer_size, &host_buffers, &events, &host_ptrs);

        if (upload_backend != null) {
            impl.logDebug("%s: using async uploads for device %s, buffer type %s, backend %s\n", .{
                "load_all_data",
                c.ggml_backend_dev_name(c.ggml_backend_get_device(upload_backend)),
                c.ggml_backend_buft_name(c.ggml_backend_buffer_get_type(bufs.at(0).?)),
                c.ggml_backend_name(upload_backend),
            });
        }

        var t: ?*c.ggml_tensor = c.ggml_get_first_tensor(ctx);
        while (t) |cur| : (t = c.ggml_get_next_tensor(ctx, cur)) {
            // This can happen with split experts models.
            const weight = self.getWeight(gguf_meta.cString(c.ggml_get_name(cur))) orelse continue;

            if (progress_callback) |cb| {
                if (!cb(@as(f32, @floatFromInt(self.size_done)) / @as(f32, @floatFromInt(self.size_data)), progress_callback_user_data)) {
                    self.freeUploads(&events, &host_buffers, upload_backend);
                    return false;
                }
            }

            const n_size = c.ggml_nbytes(cur);

            if (self.use_mmap) {
                const mapping = &self.mappings.items[weight.idx];
                const buf_mmap = bufs.at(weight.idx);
                const data: [*]u8 = @ptrFromInt(@intFromPtr(mapping.addr) + weight.offs);

                if (self.check_tensors) validation.append(allocator, .{ .tensor = cur, .data = data, .size = n_size }) catch oom();

                impl.assert(buf_mmap != null or cur.data != null, "llama-model-loader.cpp", 1570, "buf_mmap || cur->data");
                if (buf_mmap != null and cur.data == null) {
                    _ = c.ggml_backend_tensor_alloc(buf_mmap, cur, data);
                    if (lmlocks) |locks| locks.items[weight.idx].growTo(weight.offs + n_size);

                    const used = &self.mmaps_used.items[weight.idx];
                    used[0] = @min(used[0], weight.offs);
                    used[1] = @max(used[1], weight.offs + n_size);
                } else {
                    c.ggml_backend_tensor_set(cur, data, 0, n_size);
                }
            } else {
                const file = &self.files.items[weight.idx];
                if (c.ggml_backend_buffer_is_host(cur.buffer)) {
                    try file.seek(weight.offs, c.SEEK_SET, exc);
                    try file.readRaw(cur.data.?, n_size, exc);
                    if (self.check_tensors) validation.append(allocator, .{ .tensor = cur, .data = @ptrCast(cur.data.?), .size = n_size }) catch oom();
                } else if (upload_backend != null) {
                    // Load the tensor in chunks to pinned memory and upload
                    // the buffers asynchronously.
                    const offset = weight.offs;
                    alignment = file.readAlignment();
                    const aligned_offset = offset & ~(alignment - 1);
                    const offset_from_alignment = offset - aligned_offset;
                    try file.seek(aligned_offset, c.SEEK_SET, exc);

                    const read_start = aligned_offset;
                    const read_end = (offset + n_size + alignment - 1) & ~(alignment - 1);

                    var bytes_read: usize = 0;
                    var data_read: usize = 0;
                    while (bytes_read < read_end - read_start) {
                        const read_size = @min(buffer_size, read_end - read_start - bytes_read);
                        const ptr_dest_aligned = (@intFromPtr(host_ptrs.items[buffer_idx]) + alignment - 1) & ~(alignment - 1);

                        c.ggml_backend_event_synchronize(events.items[buffer_idx]);
                        try file.readRawUnsafe(@ptrFromInt(ptr_dest_aligned), read_size, exc);

                        var ptr_data = ptr_dest_aligned;
                        var data_to_copy = read_size;
                        if (bytes_read == 0) {
                            ptr_data += offset_from_alignment;
                            data_to_copy -= offset_from_alignment;
                        }
                        if (aligned_offset + bytes_read + read_size > offset + n_size) {
                            data_to_copy -= (read_end - (offset + n_size));
                        }

                        c.ggml_backend_tensor_set_async(upload_backend, cur, @ptrFromInt(ptr_data), data_read, data_to_copy);
                        c.ggml_backend_event_record(events.items[buffer_idx], upload_backend);

                        data_read += data_to_copy;
                        bytes_read += read_size;

                        buffer_idx += 1;
                        buffer_idx %= n_buffers;
                    }
                } else {
                    read_buf.resize(allocator, n_size) catch oom();
                    try file.seek(weight.offs, c.SEEK_SET, exc);
                    try file.readRaw(read_buf.items.ptr, n_size, exc);
                    c.ggml_backend_tensor_set(cur, read_buf.items.ptr, 0, n_size);
                    if (self.check_tensors and !c.ggml_validate_row_data(cur.type, read_buf.items.ptr, n_size)) {
                        return exc.throw(.runtime_error, "tensor '{s}' has invalid data", .{gguf_meta.cString(c.ggml_get_name(cur))});
                    }
                }
            }

            self.size_done += n_size;
        }

        // Free temporary resources used for async uploads.
        self.freeUploads(&events, &host_buffers, upload_backend);

        // Check validation results.
        var validation_failed = false;
        for (validation.items) |v| {
            if (!c.ggml_validate_row_data(v.tensor.type, v.data, v.size)) {
                impl.logError("%s: tensor '%s' has invalid data\n", .{ "load_all_data", c.ggml_get_name(v.tensor) });
                validation_failed = true;
            }
        }
        if (validation_failed) return exc.throw(.runtime_error, "found tensors with invalid data", .{});

        // Check if this is the last call and do final cleanup.
        if (self.size_done >= self.size_data) {
            // Unmap offloaded tensors and metadata.
            if (self.use_mmap) {
                for (self.mappings.items, 0..) |*mapping, idx| {
                    const used = self.mmaps_used.items[idx];
                    mapping.unmapFragment(0, used[0]);
                    if (used[1] != 0) mapping.unmapFragment(used[1], mapping.size);
                }
            }
            if (progress_callback) |cb| {
                // Even though the model is done loading, cancellation is
                // still honoured, since allocations need freeing.
                return cb(1.0, progress_callback_user_data);
            }
        }
        return true;
    }

    const Validation = struct { tensor: *c.ggml_tensor, data: *const anyopaque, size: usize };

    /// The `upload_backend` lambda of `load_all_data`
    /// (llama-model-loader.cpp:1426 @c1d0e7a00): a backend for async
    /// uploads from pinned memory, when nothing is mapped and the device
    /// supports it; else null.
    fn uploadBackend(
        self: *const Loader,
        bufs: *const BufMap,
        n_buffers: usize,
        buffer_size: usize,
        host_buffers: *std.ArrayList(c.ggml_backend_buffer_t),
        events: *std.ArrayList(c.ggml_backend_event_t),
        host_ptrs: *std.ArrayList(?*anyopaque),
    ) c.ggml_backend_t {
        const func = "load_all_data";
        if (self.use_mmap or self.check_tensors) return null;
        const buf = bufs.at(0) orelse {
            impl.logDebug("%s: no buffer found for async uploads\n", .{func});
            return null;
        };
        const buft = c.ggml_backend_buffer_get_type(buf);
        const dev = c.ggml_backend_buft_get_device(buft) orelse {
            impl.logDebug("%s: no device found for buffer type %s for async uploads\n", .{ func, c.ggml_backend_buft_name(buft) });
            return null;
        };
        if (buft != c.ggml_backend_dev_buffer_type(dev)) {
            impl.logDebug("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", .{ func, c.ggml_backend_buft_name(buft), c.ggml_backend_dev_name(dev) });
            return null;
        }
        var props: c.ggml_backend_dev_props = undefined;
        c.ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async or !props.caps.host_buffer or !props.caps.events) {
            impl.logDebug("%s: device %s does not support async, host buffers or events\n", .{ func, c.ggml_backend_dev_name(dev) });
            return null;
        }
        const host_buft = c.ggml_backend_dev_host_buffer_type(dev) orelse {
            impl.logDebug("%s: no host buffer type found for device %s\n", .{ func, c.ggml_backend_dev_name(dev) });
            return null;
        };
        for (0..n_buffers) |_| {
            const hbuf = c.ggml_backend_buft_alloc_buffer(host_buft, buffer_size) orelse {
                impl.logDebug("%s: failed to allocate host buffer for async uploads for device %s\n", .{ func, c.ggml_backend_dev_name(dev) });
                return null;
            };
            host_buffers.append(allocator, hbuf) catch oom();
            host_ptrs.append(allocator, c.ggml_backend_buffer_get_base(hbuf)) catch oom();
            const event = c.ggml_backend_event_new(dev) orelse {
                impl.logDebug("%s: failed to create event for async uploads for device %s\n", .{ func, c.ggml_backend_dev_name(dev) });
                return null;
            };
            events.append(allocator, event) catch oom();
        }
        const backend = c.ggml_backend_dev_init(dev, null) orelse {
            impl.logDebug("%s: failed to initialize backend for device %s for async uploads\n", .{ func, c.ggml_backend_dev_name(dev) });
            return null;
        };
        return backend;
    }

    fn freeUploads(_: *const Loader, events: *std.ArrayList(c.ggml_backend_event_t), host_buffers: *std.ArrayList(c.ggml_backend_buffer_t), upload_backend: c.ggml_backend_t) void {
        for (events.items) |event| {
            c.ggml_backend_event_synchronize(event);
            c.ggml_backend_event_free(event);
        }
        events.clearRetainingCapacity();
        for (host_buffers.items) |buf| c.ggml_backend_buffer_free(buf);
        host_buffers.clearRetainingCapacity();
        c.ggml_backend_free(upload_backend);
    }
};

// -----------------------------------------------------------------------------
// Creating tensors

/// `llama_model_loader`'s `TENSOR_NOT_REQUIRED` (llama-model-loader.h:66 @c1d0e7a00).
pub const tensor_not_required: u32 = 1 << 0;
/// `llama_model_loader`'s `TENSOR_DUPLICATED` (llama-model-loader.h:67 @c1d0e7a00).
pub const tensor_duplicated: u32 = 1 << 1;
/// `llama_model_loader`'s `TENSOR_SKIP` (llama-model-loader.h:68 @c1d0e7a00).
pub const tensor_skip: u32 = 1 << 2;
/// `llama_model_loader`'s `TENSOR_SKIP_IF_VIRTUAL` (llama-model-loader.h:69 @c1d0e7a00).
pub const tensor_skip_if_virtual: u32 = 1 << 3;
/// `llama_model_loader`'s `TENSOR_ALLOW_RESHAPE` (llama-model-loader.h:70 @c1d0e7a00).
pub const tensor_allow_reshape: u32 = 1 << 4;

/// Mirrors `LLM_TN_IMPL` (llama-arch.h:689 @c1d0e7a00): a tensor's name
/// before it is formatted.
pub const Tn = struct {
    tensor: arch.Tensor,
    suffix: ?[]const u8 = null,
    bid: i32 = -1,
    xid: i32 = -1,

    /// Return: the formatted name, `LLM_TN_IMPL`'s `str`.
    pub fn str(self: Tn) arch.Name {
        return arch.tensorName(self.tensor, self.suffix, self.bid, self.xid);
    }
};

/// One entry of `buft_list_t` (llama-model-loader.h:21 @c1d0e7a00): a device and a
/// buffer type on it.
pub const BuftEntry = struct {
    dev: c.ggml_backend_dev_t,
    buft: c.ggml_backend_buffer_type_t,
};

/// `buft_list_t`, in preference order.
pub const BuftList = std.ArrayList(BuftEntry);

/// `llama_buf_map`: the backend buffer for each file index.
pub const BufMap = struct {
    entries: std.ArrayList(struct { u32, c.ggml_backend_buffer_t }) = .empty,

    /// Return: the buffer for `idx`, or null; `count` then `at`.
    pub fn at(self: *const BufMap, idx: u32) c.ggml_backend_buffer_t {
        for (self.entries.items) |e| if (e[0] == idx) return e[1];
        return null;
    }

    /// `emplace`: keeps the first buffer given for an index.
    ///
    /// Return: nothing.
    pub fn emplace(self: *BufMap, idx: u32, buf: c.ggml_backend_buffer_t) void {
        if (self.at(idx) != null) return;
        self.entries.append(allocator, .{ idx, buf }) catch oom();
    }

    /// Return: nothing.
    pub fn deinit(self: *BufMap) void {
        self.entries.deinit(allocator);
    }
};

var warned_cpu_override = false;

fn zeroTerminate(buf: []u8, s: []const u8) [*:0]const u8 {
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return @ptrCast(buf.ptr);
}

/// Ports `weight_buft_supported` (llama-model-loader.cpp:915 @c1d0e7a00):
/// whether `dev` can run `op` on the weight `w` held in `buft`, asked with
/// a representative op node.
///
/// Return: the answer, or `error.Thrown` when no context can be made.
pub fn weightBuftSupported(hparams: *const Hparams, w: *c.ggml_tensor, op: c.enum_ggml_op, buft: c.ggml_backend_buffer_type_t, dev: c.ggml_backend_dev_t, exc: *Exception) Thrown!bool {
    if (op == c.GGML_OP_NONE) return true;

    const ctx = c.ggml_init(.{ .mem_size = c.ggml_tensor_overhead() * 8, .mem_buffer = null, .no_alloc = true }) orelse
        return exc.throw(.runtime_error, "failed to create ggml context", .{});
    defer c.ggml_free(ctx);

    const op_tensor: *c.ggml_tensor = switch (op) {
        c.GGML_OP_GET_ROWS => blk: {
            const b = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, 512);
            break :blk c.ggml_get_rows(ctx, w, b);
        },
        c.GGML_OP_MUL_MAT => blk: {
            const b = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, w.ne[0], 512, w.ne[2], w.ne[3]);
            break :blk c.ggml_mul_mat(ctx, w, b);
        },
        c.GGML_OP_MUL_MAT_ID => blk: {
            // Used for either MoE expert routing or embedded adapter routing.
            const n_ids_used: c_int = if (hparams.router_layer >= 0) 1 else @bitCast(hparams.n_expert_used);
            impl.assert(n_ids_used > 0, "llama-model-loader.cpp", 950, "n_ids_used > 0");
            const b = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, w.ne[0], n_ids_used, 512);
            const ids = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, n_ids_used, 512);
            break :blk c.ggml_mul_mat_id(ctx, w, b, ids);
        },
        c.GGML_OP_ADD => blk: {
            const a = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, w.ne[0], w.ne[1], w.ne[2], w.ne[3]);
            break :blk c.ggml_add(ctx, a, w);
        },
        c.GGML_OP_ADD_ID => blk: {
            const n_expert_used: c_int = @bitCast(hparams.n_expert_used);
            impl.assert(n_expert_used > 0, "llama-model-loader.cpp", 963, "n_expert_used > 0");
            const a = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, w.ne[0], n_expert_used, 512);
            const cc = c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_I32, n_expert_used, 512);
            break :blk c.ggml_add_id(ctx, a, w, cc);
        },
        c.GGML_OP_MUL => blk: {
            const a = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, w.ne[0], w.ne[1], w.ne[2], w.ne[3]);
            break :blk c.ggml_mul(ctx, a, w);
        },
        c.GGML_OP_DIV => blk: {
            const a = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, w.ne[0]);
            break :blk c.ggml_div(ctx, a, w);
        },
        c.GGML_OP_ROPE => blk: {
            const n_embd_head: c_int = @bitCast(hparams.nEmbdHeadV(0));
            const n_head: c_int = @bitCast(hparams.nHead(0));
            const a = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, n_embd_head, n_head, 512);
            const b = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, 512);
            break :blk c.ggml_rope_ext(ctx, a, b, w, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        },
        c.GGML_OP_SSM_CONV => blk: {
            const n_seq_tokens: i64 = 512;
            const n_seqs: i64 = 3;
            const conv_x = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, w.ne[0] - 1 + n_seq_tokens, w.ne[1], n_seqs);
            break :blk c.ggml_ssm_conv(ctx, conv_x, w);
        },
        c.GGML_OP_SSM_SCAN => blk: {
            // w is ssm_a, which is used to distinguish Mamba-1 and Mamba-2.
            const d_state: i64 = if (w.ne[0] == 1) hparams.ssm_d_state else w.ne[0];
            const n_head = w.ne[1];
            const head_dim = @divTrunc(@as(i64, hparams.ssm_d_inner), n_head);
            const n_group: i64 = if (hparams.ssm_n_group != 0) hparams.ssm_n_group else 1;
            const n_seq_tokens: i64 = 512;
            const n_seqs: i64 = 3;
            const s = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs);
            const x = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, head_dim, n_head, n_seq_tokens, n_seqs);
            const dt = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, n_head, n_seq_tokens, n_seqs);
            const B = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
            const C = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
            const ids = c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, n_seqs);
            break :blk c.ggml_ssm_scan(ctx, s, x, dt, w, B, C, ids, 1);
        },
        c.GGML_OP_RWKV_WKV6 => blk: {
            const S: i64 = 123;
            const H: i64 = 123;
            const n_tokens: i64 = 123;
            const n_seqs: i64 = 123;
            const k = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, S, H, n_tokens);
            const v = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, S, H, n_tokens);
            const r = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, S, H, n_tokens);
            const td = c.ggml_new_tensor_3d(ctx, c.GGML_TYPE_F32, S, H, n_tokens);
            const state = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, S, n_seqs, S, H);
            break :blk c.ggml_rwkv_wkv6(ctx, k, v, r, w, td, state);
        },
        c.GGML_OP_IM2COL => blk: {
            const n_embd_inp: i64 = hparams.nEmbdInp();
            const b = c.ggml_new_tensor_4d(ctx, c.GGML_TYPE_F32, n_embd_inp, w.ne[1], 1, 1);
            break :blk c.ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, c.GGML_TYPE_F16);
        },
        c.GGML_OP_SCALE => c.ggml_scale(ctx, w, 1.0),
        else => c.ggml_abort("llama-model-loader.cpp", 1041, "%s: missing test for op %s for tensor %s", "weight_buft_supported", c.ggml_op_name(op), &w.name),
    };

    // A temporary dummy buffer for the weight, so that supports_op can
    // check the buffer type.
    impl.assert(w.buffer == null, "llama-model-loader.cpp", 1045, "w->buffer == nullptr");
    w.buffer = c.ggml_backend_buft_alloc_buffer(buft, 0);
    const op_supported = c.ggml_backend_dev_supports_op(dev, op_tensor);
    c.ggml_backend_buffer_free(w.buffer);
    w.buffer = null;
    return op_supported;
}

/// Ports `select_weight_buft` (llama-model-loader.cpp:1055 @c1d0e7a00): the
/// first buffer type in the list that can use the tensor.
///
/// Return: the buffer type, null when none can, or `error.Thrown`.
pub fn selectWeightBuft(hparams: *const Hparams, tensor: *c.ggml_tensor, op: c.enum_ggml_op, buft_list: *const BuftList, exc: *Exception) Thrown!c.ggml_backend_buffer_type_t {
    impl.assert(buft_list.items.len != 0, "llama-model-loader.cpp", 1056, "!buft_list->empty()");
    for (buft_list.items) |cur| {
        if (try weightBuftSupported(hparams, tensor, op, cur.buft, cur.dev, exc)) return cur.buft;
    }
    return null;
}

/// Ports `llama_format_tensor_shape` for a shape (llama-impl.cpp:101 @c1d0e7a00).
fn formatShape(ne: []const i64, buf: *[256]u8) []const u8 {
    var len: usize = @intCast(c.snprintf(buf, buf.len, "%6lld", @as(c_longlong, ne[0])));
    for (ne[1..]) |n| {
        len += @intCast(c.snprintf(buf[len..].ptr, buf.len - len, ", %6lld", @as(c_longlong, n)));
    }
    return buf[0..len];
}

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
    std.testing.refAllDecls(Weight);
    std.testing.refAllDecls(Loader);
    std.testing.refAllDecls(Tn);
    std.testing.refAllDecls(BuftEntry);
    std.testing.refAllDecls(BufMap);
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
