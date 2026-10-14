//! `llama.h`'s top-level entry points: backend setup, capabilities, model
//! loading, the built-in chat templates, and split-file names.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama.cpp` (v0.3.0, `c1d0e7a00`): every
//! unmangled export of that file but `llama_model_save_to_file`. The loading
//! itself -- `llama_model_load_from_file_impl` and what it calls -- is
//! `model.zig`'s; the template engine is `chat.zig`'s.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const impl = @import("../impl.zig");
const loader = @import("../loader.zig");
const model_mod = @import("../model.zig");
const mmap = @import("../mmap.zig");
const chat = @import("../chat.zig");
const context = @import("../context.zig");

const c = impl.c;

extern "c" fn strncpy(dst: [*c]u8, src: [*c]const u8, n: usize) [*c]u8;
extern "c" fn strcmp(a: [*c]const u8, b: [*c]const u8) c_int;
extern "c" fn strlen(s: [*c]const u8) usize;

/// `LLAMA_VERSION`, which the C++ takes from the build
/// (`build/llamacpp.zig`'s `-DLLAMA_VERSION`): the pinned tag.
pub const version = "0.3.0";

fn oom() noreturn {
    impl.abort("llama.cpp", 0, "out of memory");
}

/// Ports `llama_flash_attn_type_name` (src/llama.cpp:37 @c1d0e7a00); the table
/// is `context.flashAttnTypeName`'s.
pub fn llama_flash_attn_type_name(flash_attn_type: c.enum_llama_flash_attn_type) callconv(.c) [*c]const u8 {
    return context.flashAttnTypeName(flash_attn_type).ptr;
}

/// Ports `llama_load_mode_name` (src/llama.cpp:49 @c1d0e7a00); the table is
/// `model.loadModeName`'s.
pub fn llama_load_mode_name(load_mode: c.enum_llama_load_mode) callconv(.c) [*c]const u8 {
    return model_mod.loadModeName(load_mode).ptr;
}

/// Ports `llama_load_mode_from_str` (src/llama.cpp:67 @c1d0e7a00). An unknown
/// name throws `std::invalid_argument`, which nothing catches.
pub fn llama_load_mode_from_str(str: [*c]const u8) callconv(.c) c.enum_llama_load_mode {
    if (strcmp(str, "auto") == 0) return c.LLAMA_LOAD_MODE_AUTO;
    if (strcmp(str, "none") == 0) return c.LLAMA_LOAD_MODE_NONE;
    if (strcmp(str, "mmap") == 0) return c.LLAMA_LOAD_MODE_MMAP;
    if (strcmp(str, "mlock") == 0) return c.LLAMA_LOAD_MODE_MLOCK;
    if (strcmp(str, "mmap+mlock") == 0) return c.LLAMA_LOAD_MODE_MMAP_MLOCK;
    if (strcmp(str, "dio") == 0) return c.LLAMA_LOAD_MODE_DIRECT_IO;
    var exc: impl.Exception = .{};
    exc.set(.invalid_argument, "unknown load mode: {s}", .{std.mem.span(@as([*:0]const u8, @ptrCast(str)))});
    exc.terminate("llama.cpp", 74);
}

/// Ports `llama_sampler_chain_default_params` (src/llama.cpp:77 @c1d0e7a00).
pub fn llama_sampler_chain_default_params() callconv(.c) c.struct_llama_sampler_chain_params {
    return .{ .no_perf = true };
}

/// Ports `llama_max_devices` (src/llama.cpp:85 @c1d0e7a00).
pub fn llama_max_devices() callconv(.c) usize {
    return 16;
}

/// Ports `llama_max_tensor_buft_overrides` (src/llama.cpp:89 @c1d0e7a00).
pub fn llama_max_tensor_buft_overrides() callconv(.c) usize {
    return 4096;
}

/// Ports `llama_supports_mmap` (src/llama.cpp:93 @c1d0e7a00).
pub fn llama_supports_mmap() callconv(.c) bool {
    return mmap.Mmap.supported;
}

/// Ports `llama_supports_mlock` (src/llama.cpp:97 @c1d0e7a00).
pub fn llama_supports_mlock() callconv(.c) bool {
    return mmap.Mlock.supported;
}

/// Ports `llama_supports_gpu_offload` (src/llama.cpp:101 @c1d0e7a00); the body
/// is `model.supportsGpuOffload`'s.
pub fn llama_supports_gpu_offload() callconv(.c) bool {
    return model_mod.supportsGpuOffload();
}

/// Ports `llama_supports_rpc` (src/llama.cpp:110 @c1d0e7a00); the body is
/// `model.supportsRpc`'s.
pub fn llama_supports_rpc() callconv(.c) bool {
    return model_mod.supportsRpc();
}

/// Ports `llama_version` (src/llama.cpp:117 @c1d0e7a00).
pub fn llama_version() callconv(.c) [*c]const u8 {
    return version;
}

/// Ports `llama_backend_init` (src/llama.cpp:121 @c1d0e7a00): ggml's clock, its
/// f16 tables through a throwaway context, and every backend.
pub fn llama_backend_init() callconv(.c) void {
    c.ggml_time_init();

    // needed to initialize f16 tables
    {
        const params: c.struct_ggml_init_params = .{ .mem_size = 0, .mem_buffer = null, .no_alloc = false };
        const ctx = c.ggml_init(params);
        c.ggml_free(ctx);
    }

    if (c.ggml_backend_reg_count() == 0) c.ggml_backend_load_all();
}

/// Ports `llama_numa_init` (src/llama.cpp:136 @c1d0e7a00): the CPU backend's
/// own NUMA setup, found by name.
pub fn llama_numa_init(numa: c.enum_ggml_numa_strategy) callconv(.c) void {
    if (numa != c.GGML_NUMA_STRATEGY_DISABLED) {
        const dev = c.ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU);
        impl.assert(dev != null, "llama.cpp", 139, "dev && \"CPU backend is not loaded\"");
        const reg = c.ggml_backend_dev_backend_reg(dev);
        const numa_init_fn: ?*const fn (c.enum_ggml_numa_strategy) callconv(.c) void =
            @ptrCast(@alignCast(c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init")));
        if (numa_init_fn) |f| f(numa);
    }
}

/// Ports `llama_backend_free` (src/llama.cpp:148 @c1d0e7a00).
pub fn llama_backend_free() callconv(.c) void {
    c.ggml_quantize_free();
}

/// Ports `llama_time_us` (src/llama.cpp:152 @c1d0e7a00).
pub fn llama_time_us() callconv(.c) i64 {
    return c.ggml_time_us();
}

/// The shared tail of the loading entry points, which the C++ spells as
/// one call to `llama_model_load_from_file_impl` (src/llama.cpp:377
/// @c1d0e7a00): exactly one source, counted as the C++ counts them, or
/// none for its error message.
///
/// Parameters:
/// - `metadata`, `path`, `file`: the candidate sources; an empty path is
///   no path.
/// - `splits`: the split paths, as the loader takes them; every element
///   is the caller's to free afterwards, those the loader adds included.
/// - `set_tensor_data`, `set_tensor_data_ud`: for a model from metadata.
/// - `params`: the model parameters.
///
/// Return: the model, or null.
fn loadImpl(
    metadata: ?*c.struct_gguf_context,
    set_tensor_data: c.llama_model_set_tensor_data_t,
    set_tensor_data_ud: ?*anyopaque,
    path: [:0]const u8,
    splits: *std.ArrayList([:0]const u8),
    file: ?*c.FILE,
    params: c.struct_llama_model_params,
) ?*c.struct_llama_model {
    var n_sources_defined: c_int = 0;
    if (metadata != null) n_sources_defined += 1;
    if (path.len != 0) n_sources_defined += 1;
    if (file != null) n_sources_defined += 1;
    const source: ?loader.Loader.Source = if (n_sources_defined != 1)
        null
    else if (metadata) |m|
        .{ .metadata = m }
    else if (file) |f|
        .{ .file = f }
    else
        .{ .path = path };
    return @ptrCast(model_mod.loadFromFileImpl(source, splits, set_tensor_data, set_tensor_data_ud, params));
}

fn freeSplits(splits: *std.ArrayList([:0]const u8)) void {
    for (splits.items) |s| impl.allocator.free(s);
    splits.deinit(impl.allocator);
}

/// Ports `llama_model_init_from_user` (src/llama.cpp:443 @c1d0e7a00): a model
/// from metadata, its tensors' data supplied by the caller, never mapped.
pub fn llama_model_init_from_user(
    metadata: ?*c.struct_gguf_context,
    set_tensor_data: c.llama_model_set_tensor_data_t,
    set_tensor_data_ud: ?*anyopaque,
    params_: c.struct_llama_model_params,
) callconv(.c) ?*c.struct_llama_model {
    impl.assert(metadata != null, "llama.cpp", 448, "metadata != nullptr");
    var params = params_;
    var splits: std.ArrayList([:0]const u8) = .empty;
    defer freeSplits(&splits);
    params.load_mode = c.LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false;
    return loadImpl(metadata, set_tensor_data, set_tensor_data_ud, "", &splits, null, params);
}

/// Ports `llama_load_model_from_file` (src/llama.cpp:456 @c1d0e7a00):
/// deprecated.
pub fn llama_load_model_from_file(path_model: [*c]const u8, params: c.struct_llama_model_params) callconv(.c) ?*c.struct_llama_model {
    return llama_model_load_from_file(path_model, params);
}

/// Ports `llama_model_load_from_file` (src/llama.cpp:462 @c1d0e7a00).
pub fn llama_model_load_from_file(path_model: [*c]const u8, params: c.struct_llama_model_params) callconv(.c) ?*c.struct_llama_model {
    var splits: std.ArrayList([:0]const u8) = .empty;
    defer freeSplits(&splits);
    return loadImpl(null, null, null, std.mem.span(@as([*:0]const u8, @ptrCast(path_model))), &splits, null, params);
}

/// Ports `llama_model_load_from_splits` (src/llama.cpp:469 @c1d0e7a00): the
/// first path is the model, and the list replaces the naming scheme.
pub fn llama_model_load_from_splits(paths: [*c][*c]const u8, n_paths: usize, params: c.struct_llama_model_params) callconv(.c) ?*c.struct_llama_model {
    if (n_paths == 0) {
        impl.logError("%s: list of splits is empty\n", .{"llama_model_load_from_splits"});
        return null;
    }
    var splits: std.ArrayList([:0]const u8) = .empty;
    defer freeSplits(&splits);
    splits.ensureTotalCapacity(impl.allocator, n_paths) catch oom();
    for (0..n_paths) |i| {
        const p = std.mem.span(@as([*:0]const u8, @ptrCast(paths[i])));
        splits.appendAssumeCapacity(impl.allocator.dupeSentinel(u8, p, 0) catch oom());
    }
    return loadImpl(null, null, null, splits.items[0], &splits, null, params);
}

/// Ports `llama_model_load_from_file_ptr` (src/llama.cpp:485 @c1d0e7a00): a
/// model read from an open file, which stays the caller's.
pub fn llama_model_load_from_file_ptr(file: [*c]c.FILE, params: c.struct_llama_model_params) callconv(.c) ?*c.struct_llama_model {
    if (file == null) {
        impl.logError("%s: file is NULL\n", .{"llama_model_load_from_file_ptr"});
        return null;
    }
    var splits: std.ArrayList([:0]const u8) = .empty;
    defer freeSplits(&splits);
    return loadImpl(null, null, null, "", &splits, file, params);
}

/// Ports `llama_chat_apply_template` (src/llama.cpp:506 @c1d0e7a00): the
/// conversation formatted with the built-in template `tmpl` names or
/// resembles -- ChatML when it is null -- and `strncpy`ed into `buf`, so
/// the copy is NUL-padded when it is shorter than `length` and not
/// terminated when it is not.
///
/// Return: the formatted length, whatever was copied; -1 for a template
/// nothing matches, with `buf` untouched.
pub fn llama_chat_apply_template(
    tmpl: [*c]const u8,
    chat_msgs: [*c]const c.struct_llama_chat_message,
    n_msg: usize,
    add_ass: bool,
    buf: [*c]u8,
    length: i32,
) callconv(.c) i32 {
    const curr_tmpl: []const u8 = if (tmpl == null) "chatml" else std.mem.span(@as([*:0]const u8, @ptrCast(tmpl)));
    const msgs: []const chat.Message = if (n_msg == 0) &.{} else chat_msgs[0..n_msg];

    var formatted_chat: std.ArrayList(u8) = .empty;
    defer formatted_chat.deinit(impl.allocator);
    const detected_tmpl = chat.detectTemplate(curr_tmpl);
    if (detected_tmpl == .unknown) return -1;
    const res = chat.applyTemplate(detected_tmpl, msgs, &formatted_chat, add_ass);
    if (res < 0) return res;
    if (buf != null and length > 0) {
        formatted_chat.append(impl.allocator, 0) catch oom();
        _ = strncpy(buf, formatted_chat.items.ptr, @intCast(length));
    }
    return res;
}

/// Ports `llama_split_path` (src/llama.cpp:541 @c1d0e7a00):
/// `<prefix>-<n>-of-<count>.gguf`, numbered from 1.
///
/// Return: the length written, or 0 when it did not fit.
pub fn llama_split_path(split_path: [*c]u8, maxlen: usize, path_prefix: [*c]const u8, split_no: i32, split_count: i32) callconv(.c) i32 {
    const written = c.snprintf(split_path, maxlen, "%s-%05d-of-%05d.gguf", path_prefix, split_no +% 1, split_count);
    if (written < 0 or @as(usize, @intCast(written)) >= maxlen) return 0;
    return written;
}

/// Ports `llama_split_prefix` (src/llama.cpp:566 @c1d0e7a00): `split_path`
/// without the postfix `llama_split_path` gives `split_no` of
/// `split_count`, copied as far as `maxlen` allows.
///
/// Return: the prefix's length, or 0 when the path does not end in that
/// postfix.
pub fn llama_split_prefix(split_prefix: [*c]u8, maxlen: usize, split_path: [*c]const u8, split_no: i32, split_count: i32) callconv(.c) i32 {
    const str_split_path = std.mem.span(@as([*:0]const u8, @ptrCast(split_path)));

    var postfix: [32]u8 = undefined;
    _ = c.snprintf(&postfix, postfix.len, "-%05d-of-%05d.gguf", split_no +% 1, split_count);
    const str_postfix = postfix[0..strlen(&postfix)];
    if (str_split_path.len <= str_postfix.len) return 0;

    const size_prefix = str_split_path.len - str_postfix.len;
    if (std.mem.eql(u8, str_split_path[size_prefix..], str_postfix)) {
        const copy_len = @min(size_prefix + 1, maxlen);
        _ = c.snprintf(split_prefix, copy_len, "%s", split_path);
        return @truncate(@as(i64, @intCast(size_prefix)));
    }
    return 0;
}

/// `llama_print_system_info`'s function-local `static std::string`.
var system_info: std.ArrayList(u8) = .empty;

/// Ports `llama_print_system_info` (src/llama.cpp:595 @c1d0e7a00): every
/// backend registry's features, as `name : key = value | ...`.
///
/// Return: the text, valid until the next call.
pub fn llama_print_system_info() callconv(.c) [*c]const u8 {
    const s = &system_info;
    s.clearRetainingCapacity();
    var i: usize = 0;
    while (i < c.ggml_backend_reg_count()) : (i += 1) {
        const reg = c.ggml_backend_reg_get(i);
        const get_features_fn: c.ggml_backend_get_features_t = @ptrCast(@alignCast(c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features")));
        if (get_features_fn) |f| {
            var features = f(reg);
            s.appendSlice(impl.allocator, std.mem.span(c.ggml_backend_reg_name(reg))) catch oom();
            s.appendSlice(impl.allocator, " : ") catch oom();
            while (features.*.name != null) : (features += 1) {
                s.appendSlice(impl.allocator, std.mem.span(features.*.name)) catch oom();
                s.appendSlice(impl.allocator, " = ") catch oom();
                s.appendSlice(impl.allocator, std.mem.span(features.*.value)) catch oom();
                s.appendSlice(impl.allocator, " | ") catch oom();
            }
        }
    }
    s.append(impl.allocator, 0) catch oom();
    return s.items.ptr;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "split names round-trip, and a short buffer gives 0" {
    var buf: [128]u8 = undefined;
    const n = llama_split_path(&buf, buf.len, "/m/x", 1, 4);
    try std.testing.expectEqualStrings("/m/x-00002-of-00004.gguf", buf[0..@intCast(n)]);
    var pre: [128]u8 = undefined;
    try std.testing.expectEqual(@as(i32, 4), llama_split_prefix(&pre, pre.len, &buf, 1, 4));
    try std.testing.expectEqualStrings("/m/x", std.mem.sliceTo(&pre, 0));
    try std.testing.expectEqual(@as(i32, 0), llama_split_prefix(&pre, pre.len, &buf, 2, 4));
    try std.testing.expectEqual(@as(i32, 0), llama_split_path(&buf, 10, "/m/x", 1, 4));
}

test "load mode names round-trip" {
    try std.testing.expectEqual(@as(c.enum_llama_load_mode, c.LLAMA_LOAD_MODE_MMAP_MLOCK), llama_load_mode_from_str("mmap+mlock"));
    try std.testing.expectEqualStrings("dio", std.mem.span(llama_load_mode_name(c.LLAMA_LOAD_MODE_DIRECT_IO)));
}
