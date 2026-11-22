//! `llama.h`'s model entry points: parameters, hyperparameters, metadata.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/src/llama-model.cpp` (v1.0.0, `b1b770d90`): the
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
const llama = @import("../module.zig");

/// The port's model behind a `llama_model *`.
fn modelOf(model: ?*const llama.impl.c.struct_llama_model) *const llama.model.Model {
    return @ptrCast(@alignCast(model.?));
}

/// `snprintf(buf, buf_size, "%s", s)` for a string with no NUL in it.
fn printTo(buf: [*c]u8, buf_size: usize, s: []const u8) i32 {
    return llama.impl.c.snprintf(buf, buf_size, "%.*s", @as(c_int, @intCast(s.len)), s.ptr);
}

/// The `i`-th entry of `gguf_kv`, in its iteration order.
fn kvAt(m: *const llama.model.Model, i: usize) struct { key: []const u8, value: [:0]u8 } {
    var it = m.gguf_kv.iterator();
    var k: usize = 0;
    while (it.next()) |e| : (k += 1) {
        if (k == i) return .{ .key = e.key, .value = e.value.* };
    }
    unreachable;
}

/// Ports `llama_model_default_params` (llama-model.cpp:3553 @b1b770d90).
pub fn llama_model_default_params() callconv(.c) llama.impl.c.struct_llama_model_params {
    return llama.model.defaultParams();
}

/// Ports `llama_model_get_vocab` (llama-model.cpp:3577 @b1b770d90): the
/// port's `Vocab` inside the model, which `api/vocab.zig` takes back.
pub fn llama_model_get_vocab(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) ?*const llama.impl.c.struct_llama_vocab {
    return @ptrCast(&modelOf(model).vocab);
}

/// Ports `llama_model_free` (llama-model.cpp:3581 @b1b770d90): `delete`,
/// which does nothing for null.
pub fn llama_model_free(model: ?*llama.impl.c.struct_llama_model) callconv(.c) void {
    const m = model orelse return;
    llama.model.destroy(@ptrCast(@alignCast(m)));
}

/// Ports `llama_model_n_ctx_train` (llama-model.cpp:3585 @b1b770d90).
pub fn llama_model_n_ctx_train(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.n_ctx_train);
}

/// Ports `llama_model_n_embd` (llama-model.cpp:3589 @b1b770d90).
pub fn llama_model_n_embd(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.n_embd);
}

/// Ports `llama_model_n_embd_inp` (llama-model.cpp:3593 @b1b770d90).
pub fn llama_model_n_embd_inp(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.nEmbdInp());
}

/// Ports `llama_model_n_embd_out` (llama-model.cpp:3597 @b1b770d90).
pub fn llama_model_n_embd_out(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.nEmbdOut());
}

/// Ports `llama_model_n_layer` (llama-model.cpp:3601 @b1b770d90).
pub fn llama_model_n_layer(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.nLayer());
}

/// Ports `llama_model_n_layer_nextn` (llama-model.cpp:3605 @b1b770d90).
pub fn llama_model_n_layer_nextn(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.n_layer_nextn);
}

/// Ports `llama_model_dflash_selector_top_k` (llama-model.cpp:3609 @b1b770d90):
/// a DFlash draft's selector width, 0 without a selector head.
pub fn llama_model_dflash_selector_top_k(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.dflash_selector_top_k);
}

/// Ports `llama_model_n_head` (llama-model.cpp:3613 @b1b770d90): layer 0's.
pub fn llama_model_n_head(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.nHead(0));
}

/// Ports `llama_model_n_head_kv` (llama-model.cpp:3617 @b1b770d90): layer 0's.
pub fn llama_model_n_head_kv(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.nHeadKv(0));
}

/// Ports `llama_model_n_swa` (llama-model.cpp:3621 @b1b770d90): DeepSeek-V4's
/// cache has a window it cannot roll back through, so it reports none.
pub fn llama_model_n_swa(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    const m = modelOf(model);
    if (m.arch == .deepseek4 or m.arch == .deepseek41) return 0;
    return @bitCast(m.hparams.n_swa);
}

/// Ports `llama_model_n_cls_out` (llama-model.cpp:3631 @b1b770d90).
pub fn llama_model_n_cls_out(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) u32 {
    return modelOf(model).hparams.n_cls_out;
}

/// Ports `llama_model_cls_label` (llama-model.cpp:3635 @b1b770d90).
///
/// Return: label `i`, borrowed from the model, or null past the last.
pub fn llama_model_cls_label(model: ?*const llama.impl.c.struct_llama_model, i: u32) callconv(.c) [*c]const u8 {
    const labels = modelOf(model).classifier_labels.items;
    if (i < labels.len) return labels[i].ptr;
    return null;
}

/// Ports `llama_model_rope_type` (llama-model.cpp:3651 @b1b770d90); the
/// table is `Model.ropeType`'s.
pub fn llama_model_rope_type(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) llama.impl.c.enum_llama_rope_type {
    return modelOf(model).ropeType();
}

/// Ports `llama_model_rope_freq_scale_train` (llama-model.cpp:3861 @b1b770d90).
pub fn llama_model_rope_freq_scale_train(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) f32 {
    return modelOf(model).hparams.rope_freq_scale_train;
}

/// Ports `llama_model_meta_val_str` (llama-model.cpp:3865 @b1b770d90).
///
/// Return: the value's length, however much of it fit; -1, with `buf`
/// emptied, for a key the model does not have.
pub fn llama_model_meta_val_str(model: ?*const llama.impl.c.struct_llama_model, key: [*c]const u8, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    const m = modelOf(model);
    const k: []const u8 = std.mem.span(@as([*:0]const u8, @ptrCast(key)));
    const v = m.gguf_kv.get(k) orelse {
        if (buf_size > 0) buf[0] = 0;
        return -1;
    };
    return printTo(buf, buf_size, v.*);
}

/// Ports `llama_model_meta_count` (llama-model.cpp:3876 @b1b770d90).
pub fn llama_model_meta_count(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @truncate(@as(i64, @intCast(modelOf(model).gguf_kv.count())));
}

/// Ports `llama_model_meta_key_str` (llama-model.cpp:3880 @b1b770d90).
///
/// Return: the key's name, static, or null for a value past the enum.
pub fn llama_model_meta_key_str(key: llama.impl.c.enum_llama_model_meta_key) callconv(.c) [*c]const u8 {
    return switch (key) {
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE => "general.sampling.sequence",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_TOP_K => "general.sampling.top_k",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_TOP_P => "general.sampling.top_p",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_MIN_P => "general.sampling.min_p",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY => "general.sampling.xtc_probability",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD => "general.sampling.xtc_threshold",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_TEMP => "general.sampling.temp",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N => "general.sampling.penalty_last_n",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT => "general.sampling.penalty_repeat",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT => "general.sampling.mirostat",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU => "general.sampling.mirostat_tau",
        llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA => "general.sampling.mirostat_eta",
        else => null,
    };
}

/// Ports `llama_model_meta_key_by_index` (llama-model.cpp:3898 @b1b770d90).
///
/// Return: the `i`-th key's length, in `gguf_kv`'s order; -1, with `buf`
/// emptied, out of range.
pub fn llama_model_meta_key_by_index(model: ?*const llama.impl.c.struct_llama_model, i: i32, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    const m = modelOf(model);
    if (i < 0 or i >= @as(i32, @truncate(@as(i64, @intCast(m.gguf_kv.count()))))) {
        if (buf_size > 0) buf[0] = 0;
        return -1;
    }
    return printTo(buf, buf_size, kvAt(m, @intCast(i)).key);
}

/// Ports `llama_model_meta_val_str_by_index` (llama-model.cpp:3910 @b1b770d90).
///
/// Return: the `i`-th value's length, in `gguf_kv`'s order; -1, with `buf`
/// emptied, out of range.
pub fn llama_model_meta_val_str_by_index(model: ?*const llama.impl.c.struct_llama_model, i: i32, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    const m = modelOf(model);
    if (i < 0 or i >= @as(i32, @truncate(@as(i64, @intCast(m.gguf_kv.count()))))) {
        if (buf_size > 0) buf[0] = 0;
        return -1;
    }
    return printTo(buf, buf_size, kvAt(m, @intCast(i)).value);
}

/// Ports `llama_model_desc` (llama-model.cpp:3922 @b1b770d90).
pub fn llama_model_desc(model: ?*const llama.impl.c.struct_llama_model, buf: [*c]u8, buf_size: usize) callconv(.c) i32 {
    return printTo(buf, buf_size, modelOf(model).desc());
}

/// Ports `llama_model_ftype` (llama-model.cpp:3926 @b1b770d90).
pub fn llama_model_ftype(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) llama.impl.c.enum_llama_ftype {
    return modelOf(model).ftype;
}

/// Ports `llama_model_size` (llama-model.cpp:3930 @b1b770d90).
pub fn llama_model_size(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) u64 {
    return modelOf(model).n_bytes;
}

/// Ports `llama_model_chat_template` (llama-model.cpp:3934 @b1b770d90):
/// `tokenizer.chat_template`, or `tokenizer.chat_template.<name>`, from the
/// metadata; with no template, the one name the C++ hard-codes for
/// Mistral-Small-2503.
///
/// Return: the template, borrowed from the model, or null.
pub fn llama_model_chat_template(model: ?*const llama.impl.c.struct_llama_model, name: [*c]const u8) callconv(.c) [*c]const u8 {
    const m = modelOf(model);
    var key: std.ArrayList(u8) = .empty;
    defer key.deinit(std.heap.c_allocator);
    const base = llama.arch.kvName(m.arch, .tokenizer_chat_template, null);
    key.appendSlice(std.heap.c_allocator, base.slice()) catch llama.impl.abort("llama-model.cpp", 3935, "out of memory");
    if (name != null) {
        key.append(std.heap.c_allocator, '.') catch llama.impl.abort("llama-model.cpp", 3935, "out of memory");
        key.appendSlice(std.heap.c_allocator, std.mem.span(@as([*:0]const u8, @ptrCast(name)))) catch llama.impl.abort("llama-model.cpp", 3935, "out of memory");
    }
    const v = m.gguf_kv.get(key.items) orelse {
        // One-off fix for very popular models: Mistral-Small-2503 has no
        // built-in chat template.
        if (name == null and m.vocab.pre_type == .tekken and m.layers.len == 40) return "mistral-v7-tekken";
        return null;
    };
    return v.*.ptr;
}

/// Ports `llama_model_n_params` (llama-model.cpp:3953 @b1b770d90).
pub fn llama_model_n_params(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) u64 {
    return modelOf(model).n_elements;
}

/// Ports `llama_model_has_encoder` (llama-model.cpp:3957 @b1b770d90).
pub fn llama_model_has_encoder(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) bool {
    return modelOf(model).hasEncoder();
}

/// Ports `llama_model_has_decoder` (llama-model.cpp:3967 @b1b770d90).
pub fn llama_model_has_decoder(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) bool {
    return modelOf(model).hasDecoder();
}

/// Ports `llama_model_decoder_start_token` (llama-model.cpp:3974 @b1b770d90).
pub fn llama_model_decoder_start_token(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) llama.impl.c.llama_token {
    return modelOf(model).hparams.dec_start_token_id;
}

/// Ports `llama_model_is_recurrent` (llama-model.cpp:3978 @b1b770d90).
pub fn llama_model_is_recurrent(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) bool {
    return llama.arch.isRecurrent(modelOf(model).arch);
}

/// Ports `llama_model_is_hybrid` (llama-model.cpp:3982 @b1b770d90).
pub fn llama_model_is_hybrid(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) bool {
    return llama.arch.isHybrid(modelOf(model).arch);
}

/// Ports `llama_model_is_diffusion` (llama-model.cpp:3986 @b1b770d90).
pub fn llama_model_is_diffusion(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) bool {
    return llama.arch.isDiffusion(modelOf(model).arch);
}

/// Ports `llama_model_n_expert` (llama-model.cpp:3994 @b1b770d90), from
/// `llama-ext.h`.
pub fn llama_model_n_expert(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @bitCast(modelOf(model).hparams.n_expert);
}

/// Ports `llama_model_n_devices` (llama-model.cpp:3998 @b1b770d90), from
/// `llama-ext.h`.
pub fn llama_model_n_devices(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) i32 {
    return @intCast(modelOf(model).devices.items.len);
}

/// Ports `llama_model_get_device` (llama-model.cpp:4002 @b1b770d90), from
/// `llama-ext.h`: the `i`-th device the model's layers may go to, or null
/// when `i` is out of range.
pub fn llama_model_get_device(model: ?*const llama.impl.c.struct_llama_model, i: c_int) callconv(.c) llama.impl.c.ggml_backend_dev_t {
    const devices = modelOf(model).devices.items;
    if (i < 0 or i >= @as(c_int, @intCast(devices.len))) return null;
    return devices[@intCast(i)].dev;
}

/// Ports `llama_model_set_attn_rot` (llama-model.cpp:4009 @b1b770d90), from
/// `llama-ext.h`: turns the KV cache attention rotation on or off for the
/// contexts created from this model afterwards; for testing.
pub fn llama_model_set_attn_rot(model: ?*llama.impl.c.struct_llama_model, value: bool) callconv(.c) void {
    const m: *llama.model.Model = @ptrCast(@alignCast(model.?));
    m.attn_rot = value;
}

/// Ports `llama_model_target_layer_ids` (llama-model.cpp:4115 @b1b770d90):
/// the target model's layers an eagle3 or DFlash draft
/// reads, or null when there are none.
pub fn llama_model_target_layer_ids(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) [*c]const i32 {
    const v = modelOf(model).target_layer_ids.items;
    return if (v.len == 0) null else v.ptr;
}

/// Ports `llama_model_target_layer_ids_n` (llama-model.cpp:4120 @b1b770d90).
pub fn llama_model_target_layer_ids_n(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) u32 {
    return @truncate(modelOf(model).target_layer_ids.items.len);
}

/// Ports `llama_tensor_rows_to_f32` (llama-model.cpp:4125 @b1b770d90): rows
/// `[row0, row0 + n_rows)` of a 2D tensor as F32.
fn tensorRowsToF32(tensor: *const llama.impl.c.ggml_tensor, row0: i64, n_rows: i64, out: [*]f32) void {
    const nelements: usize = @as(usize, @intCast(n_rows)) *% @as(usize, @intCast(tensor.ne[0]));
    const offset: usize = @as(usize, @intCast(row0)) *% tensor.nb[1];
    const nbytes: usize = llama.impl.c.ggml_row_size(tensor.type, tensor.ne[0]) *% @as(usize, @intCast(n_rows));

    if (tensor.type == llama.impl.c.GGML_TYPE_F32) {
        llama.impl.c.ggml_backend_tensor_get(tensor, out, offset, nbytes);
        return;
    }

    const buf = std.heap.c_allocator.alloc(u8, nbytes) catch llama.impl.abort("llama-model.cpp", 4135, "out of memory");
    defer std.heap.c_allocator.free(buf);
    llama.impl.c.ggml_backend_tensor_get(tensor, buf.ptr, offset, nbytes);

    const traits = llama.impl.c.ggml_get_type_traits(tensor.type);
    if (tensor.type == llama.impl.c.GGML_TYPE_F16) {
        llama.impl.c.ggml_fp16_to_fp32_row(@ptrCast(@alignCast(buf.ptr)), out, @intCast(nelements));
    } else if (tensor.type == llama.impl.c.GGML_TYPE_BF16) {
        llama.impl.c.ggml_bf16_to_fp32_row(@ptrCast(@alignCast(buf.ptr)), out, @intCast(nelements));
    } else if (llama.impl.c.ggml_is_quantized(tensor.type) and traits.*.to_float != null) {
        traits.*.to_float.?(buf.ptr, out, @intCast(nelements));
    } else {
        _ = llama.impl.c.ggml_abort("llama-model.cpp", 4146, "unsupported tensor type for dequantization: %s", llama.impl.c.ggml_type_name(tensor.type));
    }
}

/// Ports `llama_model_get_tok_embd` (llama-model.cpp:4150 @b1b770d90),
/// from `llama-ext.h`: the token embedding matrix as F32, or with `out`
/// null its element count.
pub fn llama_model_get_tok_embd(model: ?*const llama.impl.c.struct_llama_model, out: ?[*]f32) callconv(.c) u32 {
    const m = modelOf(model);
    if (m.vocab.nTokens() == 0 or m.tok_embd == null) return 0;

    const tensor = m.tok_embd.?;
    const nelements: usize = @intCast(llama.impl.c.ggml_nelements(tensor));
    if (nelements > std.math.maxInt(u32)) _ = llama.impl.c.ggml_abort("llama-model.cpp", 4157, "GGML_ASSERT(%s) failed", "nelements <= UINT32_MAX"); // for the return type

    const dst = out orelse return @intCast(nelements);

    tensorRowsToF32(tensor, 0, tensor.ne[1], dst);

    return @intCast(nelements);
}

/// Ports `llama_model_n_tokens_delay` (llama-model.cpp:4168 @b1b770d90), from
/// `llama-ext.h`: for realtime models, the number of tokens the output lags
/// behind the audio input, used by mtmd; 0 if not applicable.
pub fn llama_model_n_tokens_delay(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) u32 {
    return modelOf(model).hparams.n_tokens_delay;
}

/// Ports `llama_model_classifier_free_guidance` (llama-model.cpp:4172
/// @b1b770d90), from `llama-ext.h`: each sequence is paired with an
/// unconditional one at `n_seq_max / 2 + seq_id`, in a unified KV cache.
pub fn llama_model_classifier_free_guidance(model: ?*const llama.impl.c.struct_llama_model) callconv(.c) bool {
    return modelOf(model).hparams.classifier_free_guidance;
}

/// Ports `llama_model_get_tok_embd_rows` (llama-model.cpp:4176 @b1b770d90),
/// from `llama-ext.h`: as `llama_model_get_tok_embd`, but only the rows of
/// the given tokens, in order (`n_tokens * n_embd` elements).
///
/// Return: the number of elements written, or 0 on error.
pub fn llama_model_get_tok_embd_rows(model: ?*const llama.impl.c.struct_llama_model, tokens: ?[*]const llama.impl.c.llama_token, n_tokens: u32, out: ?[*]f32) callconv(.c) u32 {
    const m = modelOf(model);
    if (m.vocab.nTokens() == 0 or m.tok_embd == null or tokens == null or out == null) return 0;

    const tensor = m.tok_embd.?;
    const ne0: usize = @intCast(tensor.ne[0]);
    for (0..n_tokens) |i| {
        const tok = tokens.?[i];
        if (tok < 0 or tok >= tensor.ne[1]) return 0;
        tensorRowsToF32(tensor, tok, 1, out.? + i * ne0);
    }

    return n_tokens *% @as(u32, @truncate(ne0));
}

/// Ports `llama_model_decision_label` (llama-model.cpp:3643 @b1b770d90), from
/// `llama-ext.h`: the label of option `i` of a decision model
/// (`<arch>.decision.labels`), null past the last one.
pub fn llama_model_decision_label(model: ?*const llama.impl.c.struct_llama_model, i: u32) callconv(.c) [*c]const u8 {
    const labels = modelOf(model).decision_labels.items;
    if (i < labels.len) return labels[i].ptr;
    return null;
}

/// Ports `llama_arch_all` (llama-arch.cpp:1496 @b1b770d90), from
/// `llama-ext.h`. The C++ returns a `std::vector`, so this has no C signature
/// and is never exported; it is called as Zig.
///
/// Return: all the architectures known to the library, static.
pub fn llama_arch_all() []const llama.arch_table.Arch {
    return llama.arch.archAll();
}

/// Ports `llama_arch_name` (llama-arch.cpp:1500 @b1b770d90), from
/// `llama-ext.h`: the canonical name of an architecture, or "unknown".
pub fn llama_arch_name(a: llama.arch_table.Arch) [:0]const u8 {
    return llama.arch.archName(a);
}

/// Ports `llama_arch_from_string` (llama-arch.cpp:1504 @b1b770d90), from
/// `llama-ext.h`.
///
/// Return: the architecture with the given canonical name, or `.unknown`.
pub fn llama_arch_from_string(name: []const u8) llama.arch_table.Arch {
    return llama.arch.archFromString(name);
}

/// Ports `llama_arch_kv_name` (llama-arch.cpp:1508 @b1b770d90), from
/// `llama-ext.h`: the arch-resolved GGUF key of a KV id, e.g.
/// `LLM_KV_CONTEXT_LENGTH` -> `llama.context_length`.
pub fn llama_arch_kv_name(a: llama.arch_table.Arch, key: llama.arch_table.Kv) llama.arch.Name {
    return llama.arch.kvName(a, key, null);
}

/// Ports `llama_arch_supports_sm_tensor` (llama-arch.cpp:1512 @b1b770d90),
/// from `llama-ext.h`.
pub fn llama_arch_supports_sm_tensor(a: llama.arch_table.Arch) bool {
    return llama.arch.supportsSmTensor(a);
}

/// Ports `llama_arch_supports_mixed_batch` (llama-arch.cpp:1516 @b1b770d90),
/// from `llama-ext.h`.
pub fn llama_arch_supports_mixed_batch(a: llama.arch_table.Arch) bool {
    return llama.arch.supportsMixedBatch(a);
}

/// Ports `llama_arch_supports_save` (llama-arch.cpp:1480 @b1b770d90), from
/// `llama-ext.h`.
pub fn llama_arch_supports_save(a: llama.arch_table.Arch) bool {
    return llama.arch.supportsSave(a);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "meta key names stop at the enum's end" {
    try std.testing.expectEqualStrings("general.sampling.mirostat_eta", std.mem.span(llama_model_meta_key_str(llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA)));
    try std.testing.expect(llama_model_meta_key_str(llama.impl.c.LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA + 1) == null);
}
