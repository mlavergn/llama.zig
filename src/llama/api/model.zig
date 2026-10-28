//! `llama.h`'s model entry points: parameters, hyperparameters, metadata.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-model.cpp` (v0.4.0, `5266f24da`): the
//! C API at the end of the file, from `llama_model_default_params` through
//! `llama_model_is_diffusion` -- its unmangled exports. The loading entry
//! points are `llama.cpp/src/llama.cpp`'s, in `api/llama.zig`.
//!
//! Each function has the C signature and is `pub`, not `export`: the
//! in-process harnesses call these beside the C++ that defines the same
//! names, and `exports.zig` exports them once libllama swaps.
//!
//! # Strings into a caller's buffer
//!
//! The C++ writes every string with `snprintf(buf, buf_size, "%s", ...)`,
//! so the return is the full length whatever fits and the output is
//! truncated and NUL-terminated by libc. The port calls the same `snprintf`
//! with `%.*s`: the C++'s strings hold no NUL -- they are built from C
//! strings -- so the two print the same bytes and return the same length.

const std = @import("std");
const impl = @import("../impl.zig");
const arch = @import("../arch.zig");
const model_mod = @import("../model.zig");

const c = impl.c;
const Model = model_mod.Model;

/// The port's model behind a `llama_model *`.
fn modelOf(model: ?*const c.struct_llama_model) *const Model {
    return @ptrCast(@alignCast(model.?));
}

/// `snprintf(buf, buf_size, "%s", s)` for a string with no NUL in it.
fn printTo(buf: [*c]u8, buf_size: usize, s: []const u8) i32 {
    return c.snprintf(buf, buf_size, "%.*s", @as(c_int, @intCast(s.len)), s.ptr);
}

/// The `i`-th entry of `gguf_kv`, in its iteration order.
fn kvAt(m: *const Model, i: usize) struct { key: []const u8, value: [:0]u8 } {
    var it = m.gguf_kv.iterator();
    var k: usize = 0;
    while (it.next()) |e| : (k += 1) {
        if (k == i) return .{ .key = e.key, .value = e.value.* };
    }
    unreachable;
}

/// Ports `llama_model_default_params` (llama-model.cpp:2698 @5266f24da).
pub fn llama_model_default_params() callconv(.c) c.struct_llama_model_params {
    return model_mod.defaultParams();
}

/// Ports `llama_model_get_vocab` (llama-model.cpp:2722 @5266f24da): the
/// port's `Vocab` inside the model, which `api/vocab.zig` takes back.
pub fn llama_model_get_vocab(model: ?*const c.struct_llama_model) callconv(.c) ?*const c.struct_llama_vocab {
    return @ptrCast(&modelOf(model).vocab);
}

/// Ports `llama_free_model` (llama-model.cpp:2726 @5266f24da).
pub fn llama_free_model(model: ?*c.struct_llama_model) callconv(.c) void {
    llama_model_free(model);
}

/// Ports `llama_model_free` (llama-model.cpp:2730 @5266f24da): `delete`,
/// which does nothing for null.
pub fn llama_model_free(model: ?*c.struct_llama_model) callconv(.c) void {
    const m = model orelse return;
    model_mod.destroy(@ptrCast(@alignCast(m)));
}

/// Ports `llama_model_n_ctx_train` (llama-model.cpp:2734 @5266f24da).
pub fn llama_model_n_ctx_train(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.n_ctx_train);
}

/// Ports `llama_model_n_embd` (llama-model.cpp:2738 @5266f24da).
pub fn llama_model_n_embd(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.n_embd);
}

/// Ports `llama_model_n_embd_inp` (llama-model.cpp:2742 @5266f24da).
pub fn llama_model_n_embd_inp(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.nEmbdInp());
}

/// Ports `llama_model_n_embd_out` (llama-model.cpp:2746 @5266f24da).
pub fn llama_model_n_embd_out(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.nEmbdOut());
}

/// Ports `llama_model_n_layer` (llama-model.cpp:2750 @5266f24da).
pub fn llama_model_n_layer(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.nLayer());
}

/// Ports `llama_model_n_layer_nextn` (llama-model.cpp:2754 @5266f24da).
pub fn llama_model_n_layer_nextn(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.n_layer_nextn);
}

/// Ports `llama_model_dflash_selector_top_k` (llama-model.cpp:2758 @5266f24da),
/// from `llama-ext.h`: a DFlash draft's selector width. C++ linkage upstream,
/// so `exports.zig` does not export it.
pub fn llama_model_dflash_selector_top_k(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.dflash_selector_top_k);
}

/// Ports `llama_model_n_head` (llama-model.cpp:2762 @5266f24da): layer 0's.
pub fn llama_model_n_head(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.nHead(0));
}

/// Ports `llama_model_n_head_kv` (llama-model.cpp:2766 @5266f24da): layer 0's.
pub fn llama_model_n_head_kv(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.nHeadKv(0));
}

/// Ports `llama_model_n_swa` (llama-model.cpp:2770 @5266f24da): DeepSeek-V4's
/// cache has a window it cannot roll back through, so it reports none.
pub fn llama_model_n_swa(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    const m = modelOf(model);
    if (m.arch == .deepseek4) return 0;
    return @bitCast(m.hparams.n_swa);
}

/// Ports `llama_model_n_cls_out` (llama-model.cpp:2780 @5266f24da).
pub fn llama_model_n_cls_out(model: ?*const c.struct_llama_model) callconv(.c) u32 {
    return modelOf(model).hparams.n_cls_out;
}

/// Ports `llama_model_cls_label` (llama-model.cpp:2784 @5266f24da).
///
/// Return: label `i`, borrowed from the model, or null past the last.
pub fn llama_model_cls_label(model: ?*const c.struct_llama_model, i: u32) callconv(.c) [*c]const u8 {
    const labels = modelOf(model).classifier_labels.items;
    if (i < labels.len) return labels[i].ptr;
    return null;
}

/// Ports `llama_n_ctx_train` (llama-model.cpp:2793 @5266f24da).
pub fn llama_n_ctx_train(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return llama_model_n_ctx_train(model);
}

/// Ports `llama_n_embd` (llama-model.cpp:2798 @5266f24da).
pub fn llama_n_embd(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return llama_model_n_embd(model);
}

/// Ports `llama_n_layer` (llama-model.cpp:2803 @5266f24da).
pub fn llama_n_layer(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return llama_model_n_layer(model);
}

/// Ports `llama_n_head` (llama-model.cpp:2808 @5266f24da).
pub fn llama_n_head(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return llama_model_n_head(model);
}

/// Ports `llama_model_rope_type` (llama-model.cpp:2812 @5266f24da); the
/// table is `Model.ropeType`'s.
pub fn llama_model_rope_type(model: ?*const c.struct_llama_model) callconv(.c) c.enum_llama_rope_type {
    return modelOf(model).ropeType();
}

/// Ports `llama_model_rope_freq_scale_train` (llama-model.cpp:2994 @5266f24da).
pub fn llama_model_rope_freq_scale_train(model: ?*const c.struct_llama_model) callconv(.c) f32 {
    return modelOf(model).hparams.rope_freq_scale_train;
}

/// Ports `llama_model_meta_val_str` (llama-model.cpp:2998 @5266f24da).
///
/// Return: the value's length, however much of it fit; -1, with `buf`
/// emptied, for a key the model does not have.
pub fn llama_model_meta_val_str(model: ?*const c.struct_llama_model, key: [*c]const u8, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    const m = modelOf(model);
    const k: []const u8 = std.mem.span(@as([*:0]const u8, @ptrCast(key)));
    const v = m.gguf_kv.get(k) orelse {
        if (buf_size > 0) buf[0] = 0;
        return -1;
    };
    return printTo(buf, buf_size, v.*);
}

/// Ports `llama_model_meta_count` (llama-model.cpp:3009 @5266f24da).
pub fn llama_model_meta_count(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @truncate(@as(i64, @intCast(modelOf(model).gguf_kv.count())));
}

/// Ports `llama_model_meta_key_str` (llama-model.cpp:3013 @5266f24da).
///
/// Return: the key's name, static, or null for a value past the enum.
pub fn llama_model_meta_key_str(key: c.enum_llama_model_meta_key) callconv(.c) [*c]const u8 {
    return switch (key) {
        c.LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE => "general.sampling.sequence",
        c.LLAMA_MODEL_META_KEY_SAMPLING_TOP_K => "general.sampling.top_k",
        c.LLAMA_MODEL_META_KEY_SAMPLING_TOP_P => "general.sampling.top_p",
        c.LLAMA_MODEL_META_KEY_SAMPLING_MIN_P => "general.sampling.min_p",
        c.LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY => "general.sampling.xtc_probability",
        c.LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD => "general.sampling.xtc_threshold",
        c.LLAMA_MODEL_META_KEY_SAMPLING_TEMP => "general.sampling.temp",
        c.LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N => "general.sampling.penalty_last_n",
        c.LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT => "general.sampling.penalty_repeat",
        c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT => "general.sampling.mirostat",
        c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU => "general.sampling.mirostat_tau",
        c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA => "general.sampling.mirostat_eta",
        else => null,
    };
}

/// Ports `llama_model_meta_key_by_index` (llama-model.cpp:3031 @5266f24da).
///
/// Return: the `i`-th key's length, in `gguf_kv`'s order; -1, with `buf`
/// emptied, out of range.
pub fn llama_model_meta_key_by_index(model: ?*const c.struct_llama_model, i: i32, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    const m = modelOf(model);
    if (i < 0 or i >= @as(i32, @truncate(@as(i64, @intCast(m.gguf_kv.count()))))) {
        if (buf_size > 0) buf[0] = 0;
        return -1;
    }
    return printTo(buf, buf_size, kvAt(m, @intCast(i)).key);
}

/// Ports `llama_model_meta_val_str_by_index` (llama-model.cpp:3043 @5266f24da).
///
/// Return: the `i`-th value's length, in `gguf_kv`'s order; -1, with `buf`
/// emptied, out of range.
pub fn llama_model_meta_val_str_by_index(model: ?*const c.struct_llama_model, i: i32, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    const m = modelOf(model);
    if (i < 0 or i >= @as(i32, @truncate(@as(i64, @intCast(m.gguf_kv.count()))))) {
        if (buf_size > 0) buf[0] = 0;
        return -1;
    }
    return printTo(buf, buf_size, kvAt(m, @intCast(i)).value);
}

/// Ports `llama_model_desc` (llama-model.cpp:3055 @5266f24da).
pub fn llama_model_desc(model: ?*const c.struct_llama_model, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    return printTo(buf, buf_size, modelOf(model).desc());
}

/// Ports `llama_model_ftype` (llama-model.cpp:3059 @5266f24da).
pub fn llama_model_ftype(model: ?*const c.struct_llama_model) callconv(.c) c.enum_llama_ftype {
    return modelOf(model).ftype;
}

/// Ports `llama_model_size` (llama-model.cpp:3063 @5266f24da).
pub fn llama_model_size(model: ?*const c.struct_llama_model) callconv(.c) u64 {
    return modelOf(model).n_bytes;
}

/// Ports `llama_model_chat_template` (llama-model.cpp:3067 @5266f24da):
/// `tokenizer.chat_template`, or `tokenizer.chat_template.<name>`, from the
/// metadata; with no template, the one name the C++ hard-codes for
/// Mistral-Small-2503.
///
/// Return: the template, borrowed from the model, or null.
pub fn llama_model_chat_template(model: ?*const c.struct_llama_model, name: [*c]const u8) callconv(.c) [*c]const u8 {
    const m = modelOf(model);
    var key: std.ArrayList(u8) = .empty;
    defer key.deinit(impl.allocator);
    const base = arch.kvName(m.arch, .tokenizer_chat_template, null);
    key.appendSlice(impl.allocator, base.slice()) catch impl.abort("llama-model.cpp", 3068, "out of memory");
    if (name != null) {
        key.append(impl.allocator, '.') catch impl.abort("llama-model.cpp", 3068, "out of memory");
        key.appendSlice(impl.allocator, std.mem.span(@as([*:0]const u8, @ptrCast(name)))) catch impl.abort("llama-model.cpp", 3068, "out of memory");
    }
    const v = m.gguf_kv.get(key.items) orelse {
        // One-off fix for very popular models: Mistral-Small-2503 has no
        // built-in chat template.
        if (name == null and m.vocab.pre_type == .tekken and m.layers.len == 40) return "mistral-v7-tekken";
        return null;
    };
    return v.*.ptr;
}

/// Ports `llama_model_n_params` (llama-model.cpp:3086 @5266f24da).
pub fn llama_model_n_params(model: ?*const c.struct_llama_model) callconv(.c) u64 {
    return modelOf(model).n_elements;
}

/// Ports `llama_model_has_encoder` (llama-model.cpp:3090 @5266f24da).
pub fn llama_model_has_encoder(model: ?*const c.struct_llama_model) callconv(.c) bool {
    return modelOf(model).hasEncoder();
}

/// Ports `llama_model_has_decoder` (llama-model.cpp:3100 @5266f24da).
pub fn llama_model_has_decoder(model: ?*const c.struct_llama_model) callconv(.c) bool {
    return modelOf(model).hasDecoder();
}

/// Ports `llama_model_decoder_start_token` (llama-model.cpp:3107 @5266f24da).
pub fn llama_model_decoder_start_token(model: ?*const c.struct_llama_model) callconv(.c) c.llama_token {
    return modelOf(model).hparams.dec_start_token_id;
}

/// Ports `llama_model_is_recurrent` (llama-model.cpp:3111 @5266f24da).
pub fn llama_model_is_recurrent(model: ?*const c.struct_llama_model) callconv(.c) bool {
    return arch.isRecurrent(modelOf(model).arch);
}

/// Ports `llama_model_is_hybrid` (llama-model.cpp:3115 @5266f24da).
pub fn llama_model_is_hybrid(model: ?*const c.struct_llama_model) callconv(.c) bool {
    return arch.isHybrid(modelOf(model).arch);
}

/// Ports `llama_model_is_diffusion` (llama-model.cpp:3119 @5266f24da).
pub fn llama_model_is_diffusion(model: ?*const c.struct_llama_model) callconv(.c) bool {
    return arch.isDiffusion(modelOf(model).arch);
}

/// Ports `llama_model_n_expert` (llama-model.cpp:3127 @5266f24da), from
/// `llama-ext.h`.
pub fn llama_model_n_expert(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.n_expert);
}

/// Ports `llama_model_n_devices` (llama-model.cpp:3131 @5266f24da), from
/// `llama-ext.h`.
pub fn llama_model_n_devices(model: ?*const c.struct_llama_model) callconv(.c) i32 {
    return @intCast(modelOf(model).devices.items.len);
}

/// Ports `llama_model_get_device` (llama-model.cpp:3135 @5266f24da), from
/// `llama-ext.h`: the `i`-th device the model's layers may go to, or null
/// when `i` is out of range.
pub fn llama_model_get_device(model: ?*const c.struct_llama_model, i: c_int) callconv(.c) c.ggml_backend_dev_t {
    const devices = modelOf(model).devices.items;
    if (i < 0 or i >= @as(c_int, @intCast(devices.len))) return null;
    return devices[@intCast(i)].dev;
}

/// Ports `llama_model_target_layer_ids` (llama-model.cpp:3199 @5266f24da),
/// from `llama-ext.h`: the target model's layers an eagle3 or DFlash draft
/// reads, or null when there are none.
pub fn llama_model_target_layer_ids(model: ?*const c.struct_llama_model) callconv(.c) [*c]const i32 {
    const v = modelOf(model).target_layer_ids.items;
    return if (v.len == 0) null else v.ptr;
}

/// Ports `llama_model_target_layer_ids_n` (llama-model.cpp:3204 @5266f24da),
/// from `llama-ext.h`.
pub fn llama_model_target_layer_ids_n(model: ?*const c.struct_llama_model) callconv(.c) u32 {
    return @truncate(modelOf(model).target_layer_ids.items.len);
}

/// Ports `llama_model_get_tok_embd` (llama-model.cpp:3208 @5266f24da),
/// from `llama-ext.h`: the token embedding matrix as F32, or with `out`
/// null its element count.
pub fn llama_model_get_tok_embd(model: ?*const c.struct_llama_model, out: ?[*]f32) callconv(.c) u32 {
    const m = modelOf(model);
    if (m.vocab.nTokens() == 0 or m.tok_embd == null) return 0;

    const tensor = m.tok_embd.?;
    const nelements: usize = @intCast(c.ggml_nelements(tensor));
    if (nelements > std.math.maxInt(u32)) _ = c.ggml_abort("llama-model.cpp", 3215, "GGML_ASSERT(%s) failed", "nelements <= UINT32_MAX"); // for the return type

    const dst = out orelse return @intCast(nelements);

    if (tensor.type == c.GGML_TYPE_F32) {
        c.ggml_backend_tensor_get(tensor, dst, 0, nelements * @sizeOf(f32));
        return @intCast(nelements);
    }

    const buf = impl.allocator.alloc(u8, c.ggml_nbytes(tensor)) catch impl.abort("llama-model.cpp", 3226, "out of memory");
    defer impl.allocator.free(buf);
    c.ggml_backend_tensor_get(tensor, buf.ptr, 0, buf.len);

    const traits = c.ggml_get_type_traits(tensor.type);
    if (tensor.type == c.GGML_TYPE_F16) {
        c.ggml_fp16_to_fp32_row(@ptrCast(@alignCast(buf.ptr)), dst, @intCast(nelements));
    } else if (tensor.type == c.GGML_TYPE_BF16) {
        c.ggml_bf16_to_fp32_row(@ptrCast(@alignCast(buf.ptr)), dst, @intCast(nelements));
    } else if (c.ggml_is_quantized(tensor.type) and traits.*.to_float != null) {
        traits.*.to_float.?(buf.ptr, dst, @intCast(nelements));
    } else {
        _ = c.ggml_abort("llama-model.cpp", 3237, "unsupported tensor type for dequantization: %s", c.ggml_type_name(tensor.type));
    }
    return @intCast(nelements);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "meta key names stop at the enum's end" {
    try std.testing.expectEqualStrings("general.sampling.mirostat_eta", std.mem.span(llama_model_meta_key_str(c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA)));
    try std.testing.expect(llama_model_meta_key_str(c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA + 1) == null);
}
