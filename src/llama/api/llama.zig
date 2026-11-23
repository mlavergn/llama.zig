//! `llama.h`'s top-level entry points: backend setup, capabilities, model
//! loading, the built-in chat templates, and split-file names.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama.cpp` (v1.0.0, `b1b770d90`): every
//! unmangled export of that file but `llama_model_save_to_file`. The loading
//! itself -- `llama_model_load_from_file_impl` and what it calls -- is
//! `model.zig`'s; the template engine is `chat.zig`'s.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.

const std = @import("std");
const stdlib = @import("stdlib");
const llama = @import("../module.zig");

/// `LLAMA_VERSION`, which the C++ takes from the `llama-version.h` the build
/// configures: the pinned tag without its `v`. It must equal
/// `build/upstream.zig`'s `version`, which `testing/scripts/check-reference-pin`
/// asserts.
pub const version = "1.0.0";

fn oom() noreturn {
    llama.impl.abort("llama.cpp", 0, "out of memory");
}

/// Ports `llama_flash_attn_type_name` (src/llama.cpp:38 @b1b770d90); the table
/// is `context.flashAttnTypeName`'s.
pub fn llama_flash_attn_type_name(flash_attn_type: llama.impl.llama_c.enum_llama_flash_attn_type) callconv(.c) [*c]const u8 {
    return llama.context.flashAttnTypeName(flash_attn_type).ptr;
}

/// Ports `llama_load_mode_name` (src/llama.cpp:50 @b1b770d90); the table is
/// `model.loadModeName`'s.
pub fn llama_load_mode_name(load_mode: llama.impl.llama_c.enum_llama_load_mode) callconv(.c) [*c]const u8 {
    return llama.model.loadModeName(load_mode).ptr;
}

/// Ports `llama_load_mode_from_str` (src/llama.cpp:68 @b1b770d90). An unknown
/// name aborts, `GGML_ABORT`.
pub fn llama_load_mode_from_str(str: [*c]const u8) callconv(.c) llama.impl.llama_c.enum_llama_load_mode {
    if (std.mem.orderZ(u8, str, "auto") == .eq) return llama.impl.llama_c.LLAMA_LOAD_MODE_AUTO;
    if (std.mem.orderZ(u8, str, "none") == .eq) return llama.impl.llama_c.LLAMA_LOAD_MODE_NONE;
    if (std.mem.orderZ(u8, str, "mmap") == .eq) return llama.impl.llama_c.LLAMA_LOAD_MODE_MMAP;
    if (std.mem.orderZ(u8, str, "mlock") == .eq) return llama.impl.llama_c.LLAMA_LOAD_MODE_MLOCK;
    if (std.mem.orderZ(u8, str, "mmap+mlock") == .eq) return llama.impl.llama_c.LLAMA_LOAD_MODE_MMAP_MLOCK;
    if (std.mem.orderZ(u8, str, "dio") == .eq) return llama.impl.llama_c.LLAMA_LOAD_MODE_DIRECT_IO;
    llama.impl.llama_c.ggml_abort("llama.cpp", 75, "unknown load mode: %s", str);
}

/// Ports `llama_sampler_chain_default_params` (src/llama.cpp:78 @b1b770d90).
pub fn llama_sampler_chain_default_params() callconv(.c) llama.impl.llama_c.struct_llama_sampler_chain_params {
    return .{ .no_perf = true };
}

/// Ports `llama_max_devices` (src/llama.cpp:86 @b1b770d90).
pub fn llama_max_devices() callconv(.c) usize {
    return 16;
}

/// Ports `llama_max_tensor_buft_overrides` (src/llama.cpp:90 @b1b770d90).
pub fn llama_max_tensor_buft_overrides() callconv(.c) usize {
    return 4096;
}

/// Ports `llama_supports_mmap` (src/llama.cpp:94 @b1b770d90).
pub fn llama_supports_mmap() callconv(.c) bool {
    return llama.mmap.Mmap.supported;
}

/// Ports `llama_supports_mlock` (src/llama.cpp:98 @b1b770d90).
pub fn llama_supports_mlock() callconv(.c) bool {
    return llama.mmap.Mlock.supported;
}

/// Ports `llama_supports_gpu_offload` (src/llama.cpp:102 @b1b770d90); the body
/// is `model.supportsGpuOffload`'s.
pub fn llama_supports_gpu_offload() callconv(.c) bool {
    return llama.model.supportsGpuOffload();
}

/// Ports `llama_supports_rpc` (src/llama.cpp:111 @b1b770d90); the body is
/// `model.supportsRpc`'s.
pub fn llama_supports_rpc() callconv(.c) bool {
    return llama.model.supportsRpc();
}

/// Ports `llama_version` (src/llama.cpp:118 @b1b770d90).
pub fn llama_version() callconv(.c) [*c]const u8 {
    return version;
}

/// Ports `llama_backend_init` (src/llama.cpp:122 @b1b770d90): ggml's clock, its
/// f16 tables through a throwaway context, and every backend.
pub fn llama_backend_init() callconv(.c) void {
    llama.impl.llama_c.ggml_time_init();

    // needed to initialize f16 tables
    {
        const params: llama.impl.llama_c.struct_ggml_init_params = .{ .mem_size = 0, .mem_buffer = null, .no_alloc = false };
        const ctx = llama.impl.llama_c.ggml_init(params);
        llama.impl.llama_c.ggml_free(ctx);
    }

    if (llama.impl.llama_c.ggml_backend_reg_count() == 0) llama.impl.llama_c.ggml_backend_load_all();
}

/// Ports `llama_numa_init` (src/llama.cpp:137 @b1b770d90): the CPU backend's
/// own NUMA setup, found by name.
pub fn llama_numa_init(numa: llama.impl.llama_c.enum_ggml_numa_strategy) callconv(.c) void {
    if (numa != llama.impl.llama_c.GGML_NUMA_STRATEGY_DISABLED) {
        const dev = llama.impl.llama_c.ggml_backend_dev_by_type(llama.impl.llama_c.GGML_BACKEND_DEVICE_TYPE_CPU);
        llama.impl.assert(dev != null, "llama.cpp", 140, "dev && \"CPU backend is not loaded\"");
        const reg = llama.impl.llama_c.ggml_backend_dev_backend_reg(dev);
        const numa_init_fn: ?*const fn (llama.impl.llama_c.enum_ggml_numa_strategy) callconv(.c) void =
            @ptrCast(@alignCast(llama.impl.llama_c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init")));
        if (numa_init_fn) |f| f(numa);
    }
}

/// Ports `llama_backend_free` (src/llama.cpp:149 @b1b770d90).
pub fn llama_backend_free() callconv(.c) void {
    llama.impl.llama_c.ggml_quantize_free();
}

/// Ports `llama_time_us` (src/llama.cpp:153 @b1b770d90).
pub fn llama_time_us() callconv(.c) i64 {
    return llama.impl.llama_c.ggml_time_us();
}

/// The shared tail of the loading entry points, which the C++ spells as
/// one call to `llama_model_load_from_file_impl` (src/llama.cpp:379
/// @b1b770d90): exactly one source, counted as the C++ counts them, or
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
    metadata: ?*llama.impl.llama_c.struct_gguf_context,
    set_tensor_data: llama.impl.llama_c.llama_model_set_tensor_data_t,
    set_tensor_data_ud: ?*anyopaque,
    path: [:0]const u8,
    splits: *std.ArrayList([:0]const u8),
    file: ?*llama.impl.llama_c.FILE,
    params: llama.impl.llama_c.struct_llama_model_params,
) ?*llama.impl.llama_c.struct_llama_model {
    var n_sources_defined: c_int = 0;
    if (metadata != null) n_sources_defined += 1;
    if (path.len != 0) n_sources_defined += 1;
    if (file != null) n_sources_defined += 1;
    const source: ?llama.loader.Loader.Source = if (n_sources_defined != 1)
        null
    else if (metadata) |m|
        .{ .metadata = m }
    else if (file) |f|
        .{ .file = f }
    else
        .{ .path = path };
    return @ptrCast(llama.model.loadFromFileImpl(source, splits, set_tensor_data, set_tensor_data_ud, params));
}

fn freeSplits(splits: *std.ArrayList([:0]const u8)) void {
    for (splits.items) |s| std.heap.c_allocator.free(s);
    splits.deinit(std.heap.c_allocator);
}

/// Ports `llama_model_init_from_user` (src/llama.cpp:446 @b1b770d90): a model
/// from metadata, its tensors' data supplied by the caller, never mapped.
pub fn llama_model_init_from_user(
    metadata: ?*llama.impl.llama_c.struct_gguf_context,
    set_tensor_data: llama.impl.llama_c.llama_model_set_tensor_data_t,
    set_tensor_data_ud: ?*anyopaque,
    params_: llama.impl.llama_c.struct_llama_model_params,
) callconv(.c) ?*llama.impl.llama_c.struct_llama_model {
    llama.impl.assert(metadata != null, "llama.cpp", 451, "metadata != nullptr");
    var params = params_;
    var splits: std.ArrayList([:0]const u8) = .empty;
    defer freeSplits(&splits);
    params.load_mode = llama.impl.llama_c.LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false;
    return loadImpl(metadata, set_tensor_data, set_tensor_data_ud, "", &splits, null, params);
}

/// Ports `llama_model_load_from_file` (src/llama.cpp:459 @b1b770d90).
pub fn llama_model_load_from_file(path_model: [*c]const u8, params: llama.impl.llama_c.struct_llama_model_params) callconv(.c) ?*llama.impl.llama_c.struct_llama_model {
    var splits: std.ArrayList([:0]const u8) = .empty;
    defer freeSplits(&splits);
    return loadImpl(null, null, null, std.mem.span(@as([*:0]const u8, @ptrCast(path_model))), &splits, null, params);
}

/// Ports `llama_model_load_from_splits` (src/llama.cpp:466 @b1b770d90): the
/// first path is the model, and the list replaces the naming scheme.
pub fn llama_model_load_from_splits(paths: [*c][*c]const u8, n_paths: usize, params: llama.impl.llama_c.struct_llama_model_params) callconv(.c) ?*llama.impl.llama_c.struct_llama_model {
    if (n_paths == 0) {
        llama.impl.logError("%s: list of splits is empty\n", .{"llama_model_load_from_splits"});
        return null;
    }
    var splits: std.ArrayList([:0]const u8) = .empty;
    defer freeSplits(&splits);
    splits.ensureTotalCapacity(std.heap.c_allocator, n_paths) catch oom();
    for (0..n_paths) |i| {
        const p = std.mem.span(@as([*:0]const u8, @ptrCast(paths[i])));
        splits.appendAssumeCapacity(std.heap.c_allocator.dupeSentinel(u8, p, 0) catch oom());
    }
    return loadImpl(null, null, null, splits.items[0], &splits, null, params);
}

/// Ports `llama_model_load_from_file_ptr` (src/llama.cpp:482 @b1b770d90): a
/// model read from an open file, which stays the caller's.
pub fn llama_model_load_from_file_ptr(file: ?*llama.impl.llama_c.FILE, params: llama.impl.llama_c.struct_llama_model_params) callconv(.c) ?*llama.impl.llama_c.struct_llama_model {
    if (file == null) {
        llama.impl.logError("%s: file is NULL\n", .{"llama_model_load_from_file_ptr"});
        return null;
    }
    var splits: std.ArrayList([:0]const u8) = .empty;
    defer freeSplits(&splits);
    return loadImpl(null, null, null, "", &splits, file, params);
}

/// Ports `llama_chat_apply_template` (src/llama.cpp:503 @b1b770d90): the
/// conversation formatted with the built-in template `tmpl` names or
/// resembles -- ChatML when it is null -- and `strncpy`ed into `buf`, so
/// the copy is NUL-padded when it is shorter than `length` and not
/// terminated when it is not.
///
/// Return: the formatted length, whatever was copied; -1 for a template
/// nothing matches, with `buf` untouched.
pub fn llama_chat_apply_template(
    tmpl: [*c]const u8,
    chat_msgs: [*c]const llama.impl.llama_c.struct_llama_chat_message,
    n_msg: usize,
    add_ass: bool,
    buf: [*c]u8,
    length: i32,
) callconv(.c) i32 {
    const curr_tmpl: []const u8 = if (tmpl == null) "chatml" else std.mem.span(@as([*:0]const u8, @ptrCast(tmpl)));
    const msgs: []const llama.impl.llama_c.llama_chat_message = if (n_msg == 0) &.{} else chat_msgs[0..n_msg];

    var formatted_chat: std.ArrayList(u8) = .empty;
    defer formatted_chat.deinit(std.heap.c_allocator);
    const detected_tmpl = llama.chat.detectTemplate(curr_tmpl);
    if (detected_tmpl == .unknown) return -1;
    const res = llama.chat.applyTemplate(detected_tmpl, msgs, &formatted_chat, add_ass);
    if (res < 0) return res;
    if (buf != null and length > 0) {
        formatted_chat.append(std.heap.c_allocator, 0) catch oom();
        // `strncpy`: up to `length` bytes, stopping at the terminator, the
        // rest zero-filled.
        const n: usize = @intCast(length);
        const src = formatted_chat.items;
        var i: usize = 0;
        while (i < n and src[i] != 0) : (i += 1) buf[i] = src[i];
        @memset(buf[i..n], 0);
    }
    return res;
}

/// Ports `llama_split_path` (src/llama.cpp:538 @b1b770d90):
/// `<prefix>-<n>-of-<count>.gguf`, numbered from 1.
///
/// Return: the length written, or 0 when it did not fit.
pub fn llama_split_path(split_path: [*c]u8, maxlen: usize, path_prefix: [*c]const u8, split_no: i32, split_count: i32) callconv(.c) i32 {
    const written = stdlib.libc.snprintf(split_path, maxlen, "%s-%05d-of-%05d.gguf", path_prefix, split_no +% 1, split_count);
    if (written < 0 or @as(usize, @intCast(written)) >= maxlen) return 0;
    return written;
}

/// Ports `llama_split_prefix` (src/llama.cpp:563 @b1b770d90): `split_path`
/// without the postfix `llama_split_path` gives `split_no` of
/// `split_count`, copied as far as `maxlen` allows.
///
/// Return: the prefix's length, or 0 when the path does not end in that
/// postfix.
pub fn llama_split_prefix(split_prefix: [*c]u8, maxlen: usize, split_path: [*c]const u8, split_no: i32, split_count: i32) callconv(.c) i32 {
    const str_split_path = std.mem.span(@as([*:0]const u8, @ptrCast(split_path)));

    var postfix: [32]u8 = undefined;
    _ = stdlib.libc.snprintf(&postfix, postfix.len, "-%05d-of-%05d.gguf", split_no +% 1, split_count);
    const str_postfix = std.mem.sliceTo(&postfix, 0);
    if (str_split_path.len <= str_postfix.len) return 0;

    const size_prefix = str_split_path.len - str_postfix.len;
    if (std.mem.eql(u8, str_split_path[size_prefix..], str_postfix)) {
        const copy_len = @min(size_prefix + 1, maxlen);
        _ = stdlib.libc.snprintf(split_prefix, copy_len, "%s", split_path);
        return @truncate(@as(i64, @intCast(size_prefix)));
    }
    return 0;
}

/// `llama_print_system_info`'s function-local `static std::string`.
var system_info: std.ArrayList(u8) = .empty;

/// Ports `llama_print_system_info` (src/llama.cpp:592 @b1b770d90): every
/// backend registry's features, as `name : key = value | ...`.
///
/// Return: the text, valid until the next call.
pub fn llama_print_system_info() callconv(.c) [*c]const u8 {
    const s = &system_info;
    s.clearRetainingCapacity();
    var i: usize = 0;
    while (i < llama.impl.llama_c.ggml_backend_reg_count()) : (i += 1) {
        const reg = llama.impl.llama_c.ggml_backend_reg_get(i);
        const get_features_fn: llama.impl.llama_c.ggml_backend_get_features_t = @ptrCast(@alignCast(llama.impl.llama_c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features")));
        if (get_features_fn) |f| {
            var features = f(reg);
            s.appendSlice(std.heap.c_allocator, std.mem.span(llama.impl.llama_c.ggml_backend_reg_name(reg))) catch oom();
            s.appendSlice(std.heap.c_allocator, " : ") catch oom();
            while (features.*.name != null) : (features += 1) {
                s.appendSlice(std.heap.c_allocator, std.mem.span(features.*.name)) catch oom();
                s.appendSlice(std.heap.c_allocator, " = ") catch oom();
                s.appendSlice(std.heap.c_allocator, std.mem.span(features.*.value)) catch oom();
                s.appendSlice(std.heap.c_allocator, " | ") catch oom();
            }
        }
    }
    s.append(std.heap.c_allocator, 0) catch oom();
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
    try std.testing.expectEqual(@as(llama.impl.llama_c.enum_llama_load_mode, llama.impl.llama_c.LLAMA_LOAD_MODE_MMAP_MLOCK), llama_load_mode_from_str("mmap+mlock"));
    try std.testing.expectEqualStrings("dio", std.mem.span(llama_load_mode_name(llama.impl.llama_c.LLAMA_LOAD_MODE_DIRECT_IO)));
}
