//! The clip context: loading a projector from GGUF, building and running
//! its encoder graph.
//!
//! # Provenance
//!
//! Ports `clip_ctx`, `clip_model_loader` and the free functions of
//! `llama.cpp/tools/mtmd/clip.cpp` at v0.4.1 (`b29c606e2`), apart from the
//! four per-projector switches, which live beside it:
//!
//! - `arms_hparams.zig` -- `load_hparams`'s model-specific switch
//! - `arms_tensors.zig` -- `load_tensors`'s model-specific switch
//! - `arms_inputs.zig` -- `clip_encode`'s per-projector input switch and
//!   `clip_n_output_tokens`
//! - `models/module.zig` -- `clip_get_graph_builder` and the graphs
//!
//! Each declaration names what it ports and the line it began at.
//!
//! # Exceptions
//!
//! Upstream's loader throws `std::runtime_error`, and `clip_init` catches it
//! and logs `what()`. Here a throw is `throwErr`, which records the message
//! in `what_buf` and returns `error.Runtime`; `init` logs it the same way.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("impl.zig");
const names = @import("names.zig");
const model_mod = @import("model.zig");
const graph_mod = @import("graph.zig");
const arms_hparams = @import("arms_hparams.zig");
const arms_tensors = @import("arms_tensors.zig");
const arms_inputs = @import("arms_inputs.zig");
const models = @import("models/module.zig");
const libcxx = @import("llama").libcxx;
const c = impl.c;
const gpa = impl.gpa;
const T = graph_mod.T;

pub const Error = error{ Runtime, OutOfMemory };

/// `e.what()` of the last throw.
pub var what_buf: [2048]u8 = undefined;
pub var what_len: usize = 0;

extern fn snprintf(buf: ?[*]u8, size: usize, format: [*:0]const u8, ...) c_int;

/// `throw std::runtime_error(string_format(fmt, args...))`.
pub fn throwErr(comptime fmt: [*:0]const u8, args: anytype) Error {
    const n = @call(.auto, snprintf, .{ &what_buf, what_buf.len, fmt } ++ args);
    what_len = @min(@as(usize, @intCast(@max(n, 0))), what_buf.len - 1);
    return error.Runtime;
}

/// `throw std::runtime_error(s)` for a message already built.
pub fn throwStr(s: []const u8) Error {
    what_len = @min(s.len, what_buf.len - 1);
    @memcpy(what_buf[0..what_len], s[0..what_len]);
    return error.Runtime;
}

pub fn what() [:0]const u8 {
    what_buf[what_len] = 0;
    return what_buf[0..what_len :0];
}

/// Ports `clip_context_params` (clip.h:49 @b29c606e2).
pub const ContextParams = extern struct {
    use_gpu: bool,
    device: c.ggml_backend_dev_t,
    flash_attn_type: impl.FlashAttnType,
    image_min_tokens: c_int,
    image_max_tokens: c_int,
    warmup: bool,
    cb_eval: c.ggml_backend_sched_eval_callback,
    cb_eval_user_data: ?*anyopaque,
    no_alloc: bool,
    progress_callback: ?*const fn (f32, ?*anyopaque) callconv(.c) bool,
    progress_callback_user_data: ?*anyopaque,
};

/// Ports `clip_init_result` (clip.h:63 @b29c606e2).
pub const InitResult = struct {
    ctx_v: ?*Ctx = null, // vision context
    ctx_a: ?*Ctx = null, // audio context
    ctx_gen_a: ?*Ctx = null, // audio generation context
};

/// Ports `clip_cap` (clip.h:135 @b29c606e2).
pub const Cap = struct { has_vision: bool, has_audio: bool };

/// A `std::map<ggml_backend_dev_t, size_t>`: by device pointer, ascending.
pub const DevSizeMap = struct {
    items: std.ArrayList(struct { dev: c.ggml_backend_dev_t, size: usize }) = .empty,

    pub fn add(self: *DevSizeMap, dev: c.ggml_backend_dev_t, n: usize) !void {
        const key = @intFromPtr(dev);
        var i: usize = 0;
        while (i < self.items.items.len) : (i += 1) {
            const k = @intFromPtr(self.items.items[i].dev);
            if (k == key) {
                self.items.items[i].size += n;
                return;
            }
            if (k > key) break;
        }
        try self.items.insert(gpa, i, .{ .dev = dev, .size = n });
    }

    pub fn clear(self: *DevSizeMap) void {
        self.items.clearRetainingCapacity();
    }

    pub fn deinit(self: *DevSizeMap) void {
        self.items.deinit(gpa);
    }
};

/// Ports `clip_encode_params` (clip.h:95 @b29c606e2).
pub const EncodeParams = struct {
    n_threads: c_int = 1,
    imgs: ?*const impl.ImageF32Batch = null,
    out_embd: ?*std.ArrayList(f32) = null,

    // for audio gen, imgs has exactly one entry: hidden state from backbone (GEN_CODE) or unused (GEN_WAV)
    gen_process: impl.GenProcessType = .unknown,

    // GEN_CODE: out_embd receives the embd to feed back to the backbone
    code0: i32 = 0, // semantic code sampled by the backbone
    top_k: i32 = 50,
    top_p: f32 = 1.0,
    out_codes: ?*std.ArrayList(i32) = null, // this frame's 16 sampled codes
    out_feats: ?*std.ArrayList(f32) = null, // continuous counterpart of out_codes
    seed: u32 = std.math.maxInt(u32), // UINT32_MAX for random
    temp: f32 = 0.0, // sampling temperature, noise scale for flow-matching decoders
    out_is_eos: ?*bool = null,

    // GEN_WAV
    codes: ?*const std.ArrayList(i32) = null, // this frame's 16 RVQ codes
    feats: ?*const std.ArrayList(f32) = null, // continuous counterpart of codes
    out_audio: ?*std.ArrayList(f32) = null, // decoded PCM samples, F32
    state_in: ?*const std.ArrayList(u8) = null, // state from previous call, null or wrong size means cold start
    state_out: ?*std.ArrayList(u8) = null, // state for the next call
};

/// Ports `clip_ctx` (mtmd/clip.cpp:147 @b29c606e2).
pub const Ctx = struct {
    model: model_mod.Model = .{},

    ctx_gguf: ?*c.gguf_context = null,
    ctx_data: ?*c.ggml_context = null,

    buf_compute_meta: std.ArrayList(u8) = .empty,

    backend_ptrs: std.ArrayList(c.ggml_backend_t) = .empty,
    backend_buft: std.ArrayList(c.ggml_backend_buffer_type_t) = .empty,

    backend: c.ggml_backend_t = null,
    backend_cpu: c.ggml_backend_t = null,
    buf: c.ggml_backend_buffer_t = null,

    max_nodes: c_int = 8192,
    sched: c.ggml_backend_sched_t = null,
    flash_attn_type: impl.FlashAttnType = .auto,
    is_allocated: bool = false,

    debug_output_embeddings: bool = false,

    // for measuring memory usage
    no_alloc: bool = false,
    mem_usage: DevSizeMap = .{},
    mem_compute: DevSizeMap = .{},

    support_batch: bool = false,

    // for audio gen, reseeded only when the caller asks for another seed
    rng: libcxx.Mt19937 = undefined,
    rng_seed: u32 = std.math.maxInt(u32),

    /// Ports `clip_ctx` (mtmd/clip.cpp:181 @b29c606e2), a member of `clip_ctx`.
    pub fn create(ctx_params: *const ContextParams) Error!*Ctx {
        const self = try gpa.create(Ctx);
        self.* = .{};
        self.rng = libcxx.Mt19937.init(randomDevice());
        errdefer {
            self.destroy();
        }
        self.flash_attn_type = ctx_params.flash_attn_type;
        self.no_alloc = ctx_params.no_alloc;
        self.backend_cpu = c.ggml_backend_init_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU, null);
        if (self.backend_cpu == null) return throwErr("failed to initialize CPU backend", .{});
        if (ctx_params.use_gpu) {
            if (ctx_params.device != null) {
                self.backend = c.ggml_backend_dev_init(ctx_params.device, null);
                if (self.backend == null) {
                    return throwErr("%s: failed to initialize \"%s\" backend\n", .{ "clip_ctx", c.ggml_backend_dev_name(ctx_params.device) });
                }
            } else {
                self.backend = c.ggml_backend_init_by_type(c.GGML_BACKEND_DEVICE_TYPE_GPU, null);
                if (self.backend == null) self.backend = c.ggml_backend_init_by_type(c.GGML_BACKEND_DEVICE_TYPE_IGPU, null);
            }
        }

        if (self.backend != null) {
            impl.logInf("%s: CLIP using %s backend\n", .{ "clip_ctx", c.ggml_backend_name(self.backend) });
            try self.backend_ptrs.append(gpa, self.backend);
            try self.backend_buft.append(gpa, c.ggml_backend_get_default_buffer_type(self.backend));
        } else {
            self.backend = self.backend_cpu;
            impl.logInf("%s: CLIP using CPU backend\n", .{"clip_ctx"});
        }

        if (ctx_params.image_min_tokens > 0) self.model.hparams.custom_image_min_tokens = ctx_params.image_min_tokens;
        if (ctx_params.image_max_tokens > 0) self.model.hparams.custom_image_max_tokens = ctx_params.image_max_tokens;

        try self.backend_ptrs.append(gpa, self.backend_cpu);
        try self.backend_buft.append(gpa, c.ggml_backend_get_default_buffer_type(self.backend_cpu));

        self.sched = c.ggml_backend_sched_new(self.backend_ptrs.items.ptr, self.backend_buft.items.ptr, @intCast(self.backend_ptrs.items.len), 8192, false, true);

        if (ctx_params.cb_eval != null) c.ggml_backend_sched_set_eval_callback(self.sched, ctx_params.cb_eval, ctx_params.cb_eval_user_data);

        self.debug_output_embeddings = std.c.getenv("MTMD_DEBUG_EMBEDDINGS") != null;
        return self;
    }

    /// Ports `clip_ctx` (mtmd/clip.cpp:231 @b29c606e2), its destructor, with the smart
    /// pointers' destructors in reverse member order.
    pub fn destroy(self: *Ctx) void {
        c.ggml_backend_free(self.backend);
        if (self.backend != self.backend_cpu) c.ggml_backend_free(self.backend_cpu);
        if (self.sched != null) c.ggml_backend_sched_free(self.sched);
        if (self.buf != null) c.ggml_backend_buffer_free(self.buf);
        self.backend_buft.deinit(gpa);
        self.backend_ptrs.deinit(gpa);
        self.buf_compute_meta.deinit(gpa);
        if (self.ctx_data != null) c.ggml_free(self.ctx_data);
        if (self.ctx_gguf != null) c.gguf_free(self.ctx_gguf);
        self.mem_usage.deinit();
        self.mem_compute.deinit();
        self.model.hparams.deinit();
        gpa.destroy(self);
    }

    pub fn projType(self: *const Ctx) names.ProjectorType {
        return self.model.proj_type;
    }

    /// What a `graph.Graph` is built from.
    pub fn graphSource(self: *Ctx) graph_mod.Source {
        return .{
            .model = &self.model,
            .buf_compute_meta = self.buf_compute_meta.items,
            .max_nodes = self.max_nodes,
            .flash_attn_type = self.flash_attn_type,
            .n_mmproj_embd = nMmprojEmbd(self),
        };
    }
};

/// `std::random_device{}()`: libc++ reads `arc4random` on Apple platforms,
/// `rand_s` on Windows and `getentropy` elsewhere; either way the seed is not
/// reproducible.
fn randomDevice() u32 {
    var v: u32 = 0;
    switch (comptime builtin.os.tag) {
        .macos, .ios, .tvos, .watchos, .visionos => std.c.arc4random_buf(@ptrCast(&v), @sizeOf(u32)),
        .windows => {
            const rand_s = @extern(*const fn (*c_uint) callconv(.c) c_int, .{ .name = "rand_s" });
            var r: c_uint = 0;
            _ = rand_s(&r);
            v = r;
        },
        else => {
            const getentropy = @extern(*const fn (*anyopaque, usize) callconv(.c) c_int, .{ .name = "getentropy" });
            _ = getentropy(@ptrCast(&v), @sizeOf(u32));
        },
    }
    return v;
}

// -----------------------------------------------------------------------------
// clip_model_loader

/// Ports `clip_model_loader` (mtmd/clip.cpp:1158 @b29c606e2).
pub const Loader = struct {
    ctx_meta: ?*c.ggml_context = null,
    ctx_gguf: ?*c.gguf_context = null,

    fname: [:0]const u8,

    model_size: usize = 0, // in bytes

    has_vision: bool = false,
    has_audio: bool = false,
    has_gen_audio: bool = false,

    progress_callback: ?*const fn (f32, ?*anyopaque) callconv(.c) bool = null,
    progress_callback_user_data: ?*anyopaque = null,

    /// Ports `clip_model_loader` (mtmd/clip.cpp:1174 @b29c606e2), a member of `clip_model_loader`.
    pub fn init(fname: [:0]const u8, skip_tensors: bool, progress_cb: ?*const fn (f32, ?*anyopaque) callconv(.c) bool, progress_user_data: ?*anyopaque) Error!Loader {
        const func = "clip_model_loader";
        var self: Loader = .{ .fname = fname, .progress_callback = progress_cb, .progress_callback_user_data = progress_user_data };
        var meta: ?*c.ggml_context = null;
        const params: c.gguf_init_params = .{ .no_alloc = true, .ctx = &meta };

        self.ctx_gguf = c.gguf_init_from_file(fname.ptr, params);
        if (self.ctx_gguf == null) return throwErr("%s: failed to load CLIP model from %s. Does this file exist?\n", .{ func, fname.ptr });

        self.ctx_meta = meta;

        const n_tensors: c_int = @intCast(c.gguf_get_n_tensors(self.ctx_gguf));

        // print gguf info
        {
            var name_buf: ?[*:0]const u8 = "";
            var desc_buf: ?[*:0]const u8 = "";
            try self.getStringPtr(names.KEY_NAME, &name_buf, false);
            try self.getStringPtr(names.KEY_DESCRIPTION, &desc_buf, false);
            impl.logInf("%s: model name:   %s\n", .{ func, name_buf.? });
            impl.logInf("%s: description:  %s\n", .{ func, desc_buf.? });
            impl.logInf("%s: GGUF version: %d\n", .{ func, @as(c_int, @intCast(c.gguf_get_version(self.ctx_gguf))) });
            impl.logInf("%s: alignment:    %zu\n", .{ func, c.gguf_get_alignment(self.ctx_gguf) });
            impl.logInf("%s: n_tensors:    %d\n", .{ func, n_tensors });
            impl.logInf("%s: n_kv:         %d\n", .{ func, @as(c_int, @intCast(c.gguf_get_n_kv(self.ctx_gguf))) });
            impl.logInf("\n", .{});
        }

        // modalities
        {
            try self.getBool(names.KEY_HAS_VISION_ENC, &self.has_vision, false);
            try self.getBool(names.KEY_HAS_AUDIO_ENC, &self.has_audio, false);
            try self.getBool(names.KEY_HAS_GEN_AUDIO_ENC, &self.has_gen_audio, false);

            if (self.has_vision) impl.logInf("%s: has vision encoder\n", .{func});
            if (self.has_audio) impl.logInf("%s: has audio encoder\n", .{func});
            if (self.has_gen_audio) impl.logInf("%s: has audio generation (gen) encoder\n", .{func});
        }

        // tensors
        if (!skip_tensors) {
            var i: c_int = 0;
            while (i < n_tensors) : (i += 1) {
                const tname = c.gguf_get_tensor_name(self.ctx_gguf, i);
                const offset = c.gguf_get_tensor_offset(self.ctx_gguf, i);
                const ttype = c.gguf_get_tensor_type(self.ctx_gguf, i);
                const cur = c.ggml_get_tensor(meta, tname);
                const tensor_size = c.ggml_nbytes(cur);
                self.model_size += tensor_size;
                impl.logDbg("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%llu, %llu, %llu, %llu], type = %s\n", .{
                    func,                                    i,                                       c.ggml_n_dims(cur),                      @as([*:0]const u8, @ptrCast(&cur.*.name)),
                    tensor_size,                             offset,                                  @as(c_ulonglong, @bitCast(cur.*.ne[0])), @as(c_ulonglong, @bitCast(cur.*.ne[1])),
                    @as(c_ulonglong, @bitCast(cur.*.ne[2])), @as(c_ulonglong, @bitCast(cur.*.ne[3])), c.ggml_type_name(ttype),
                });
            }
        }
        return self;
    }

    /// The smart pointers' destructors.
    pub fn deinit(self: *Loader) void {
        if (self.ctx_gguf != null) c.gguf_free(self.ctx_gguf);
        if (self.ctx_meta != null) c.ggml_free(self.ctx_meta);
    }

    fn findKey(self: *const Loader, key: [*:0]const u8) i64 {
        return c.gguf_find_key(self.ctx_gguf, key);
    }

    /// Ports `get_bool` (mtmd/clip.cpp:3807 @b29c606e2), a member of `clip_model_loader`.
    pub fn getBool(self: *const Loader, key: [*:0]const u8, output: *bool, required: bool) Error!void {
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        output.* = c.gguf_get_val_bool(self.ctx_gguf, i);
    }

    /// Ports `get_i32` (mtmd/clip.cpp:3818 @b29c606e2), a member of `clip_model_loader`.
    pub fn getI32(self: *const Loader, key: [*:0]const u8, output: *i32, required: bool) Error!void {
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        output.* = c.gguf_get_val_i32(self.ctx_gguf, i);
    }

    /// Ports `get_u32` (mtmd/clip.cpp:3829 @b29c606e2), a member of `clip_model_loader`.
    pub fn getU32(self: *const Loader, key: [*:0]const u8, output: *i32, required: bool) Error!void {
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        const val = c.gguf_get_val_u32(self.ctx_gguf, i);
        // sanity check
        if (val > @as(u32, std.math.maxInt(i32))) {
            return throwErr("%s: value %u for key '%s' exceeds INT32_MAX\n", .{ "get_u32", val, key });
        }
        output.* = @intCast(val);
    }

    /// Ports `get_f32` (mtmd/clip.cpp:3846 @b29c606e2), a member of `clip_model_loader`.
    pub fn getF32(self: *const Loader, key: [*:0]const u8, output: *f32, required: bool) Error!void {
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        output.* = c.gguf_get_val_f32(self.ctx_gguf, i);
    }

    /// Ports `get_arr_f32` (mtmd/clip.cpp:3857 @b29c606e2), a member of `clip_model_loader`.
    pub fn getArrF32(self: *const Loader, key: [*:0]const u8, output: *std.ArrayList(f32), required: bool) Error!void {
        const func = "get_arr_f32";
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        if (c.gguf_get_kv_type(self.ctx_gguf, i) != c.GGUF_TYPE_ARRAY) return throwErr("%s: key '%s' is not an array\n", .{ func, key });
        const t = c.gguf_get_arr_type(self.ctx_gguf, i);
        if (t != c.GGUF_TYPE_FLOAT32) return throwErr("%s: array '%s' has type %d, expected %d (GGUF_TYPE_FLOAT32)\n", .{ func, key, @as(c_int, @intCast(t)), @as(c_int, c.GGUF_TYPE_FLOAT32) });
        const n = c.gguf_get_arr_n(self.ctx_gguf, i);
        if (n > @as(usize, std.math.maxInt(c_int))) return throwErr("%s: array '%s' is too large (%zu elements)\n", .{ func, key, n });
        try output.resize(gpa, n);
        const values: [*]align(1) const f32 = @ptrCast(c.gguf_get_arr_data(self.ctx_gguf, i));
        for (0..n) |j| output.items[j] = values[j];
    }

    /// Ports `get_string` (mtmd/clip.cpp:3883 @b29c606e2), a member of `clip_model_loader`:
    /// the value, borrowed from the GGUF context, or `output` left alone.
    pub fn getStringPtr(self: *const Loader, key: [*:0]const u8, output: *?[*:0]const u8, required: bool) Error!void {
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        output.* = c.gguf_get_val_str(self.ctx_gguf, i);
    }

    /// `get_string` into an owned copy, the C++'s `std::string` output.
    /// The copy keeps a NUL past its end, so `c_str()` is `output.ptr`.
    pub fn getString(self: *const Loader, key: [*:0]const u8, output: *[]const u8, required: bool) Error!void {
        var p: ?[*:0]const u8 = null;
        try self.getStringPtr(key, &p, required);
        if (p) |s| output.* = try gpa.dupeSentinel(u8, std.mem.span(s), 0);
    }

    /// Ports `get_arr_int` (mtmd/clip.cpp:3894 @b29c606e2), a member of `clip_model_loader`.
    pub fn getArrInt(self: *const Loader, key: [*:0]const u8, output: *std.ArrayList(i32), required: bool) Error!void {
        const func = "get_arr_int";
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        if (c.gguf_get_kv_type(self.ctx_gguf, i) != c.GGUF_TYPE_ARRAY) return throwErr("%s: key '%s' is not an array\n", .{ func, key });
        const t = c.gguf_get_arr_type(self.ctx_gguf, i);
        if (t != c.GGUF_TYPE_INT32) return throwErr("%s: array '%s' has type %d, expected %d (GGUF_TYPE_INT32)\n", .{ func, key, @as(c_int, @intCast(t)), @as(c_int, c.GGUF_TYPE_INT32) });
        const n = c.gguf_get_arr_n(self.ctx_gguf, i);
        if (n > @as(usize, std.math.maxInt(c_int))) return throwErr("%s: array '%s' is too large (%zu elements)\n", .{ func, key, n });
        try output.resize(gpa, n);
        const values: [*]align(1) const i32 = @ptrCast(c.gguf_get_arr_data(self.ctx_gguf, i));
        for (0..n) |j| output.items[j] = values[j];
    }

    fn keyNotFound(key: [*:0]const u8) Error {
        return throwErr("Key not found: %s", .{key});
    }

    /// Ports `set_llava_uhd_res_candidates` (mtmd/clip.cpp:3920 @b29c606e2), a member of `clip_model_loader`.
    pub fn setLlavaUhdResCandidates(model: *model_mod.Model, max_patches_per_side: c_int) Error!void {
        const hparams = &model.hparams;
        var x: c_int = 1;
        while (x <= max_patches_per_side) : (x += 1) {
            var y: c_int = 1;
            while (y <= max_patches_per_side) : (y += 1) {
                if (x == 1 and y == 1) continue; // skip the first point
                try hparams.image_res_candidates.append(gpa, .{ .width = x * hparams.image_size, .height = y * hparams.image_size });
            }
        }
    }

    /// Ports `set_internvl_dhr_res_candidates` (mtmd/clip.cpp:3935 @b29c606e2), a member of `clip_model_loader`.
    pub fn setInternvlDhrResCandidates(model: *model_mod.Model) Error!void {
        const hparams = &model.hparams;
        const min_num = hparams.preproc_min_tiles;
        const max_num = hparams.preproc_max_tiles;
        if (min_num < 1) return; // avoid  divide by 0
        var a = min_num;
        while (a <= max_num) : (a += 1) {
            var b_lo = @divTrunc(min_num + a - 1, a);
            var b_hi = @divTrunc(max_num, a);
            b_lo = @max(b_lo, min_num);
            b_hi = @min(b_hi, max_num);
            var b = b_lo;
            while (b <= b_hi) : (b += 1) {
                try hparams.image_res_candidates.append(gpa, .{ .width = a * hparams.image_size, .height = b * hparams.image_size });
            }
        }
    }

    /// Ports `load_hparams` (mtmd/clip.cpp:1244 @b29c606e2), a member of `clip_model_loader`;
    /// the model-specific switch is `arms_hparams.load`.
    pub fn loadHparams(self: *Loader, model: *model_mod.Model, modality: impl.Modality) Error!void {
        const func = "load_hparams";
        const hparams = &model.hparams;
        var log_ffn_op: [*:0]const u8 = ""; // for logging

        // sanity check
        switch (modality) {
            .vision => impl.ggmlAssert(self.has_vision, "clip.cpp", 1250, "has_vision"),
            .audio => impl.ggmlAssert(self.has_audio, "clip.cpp", 1252, "has_audio"),
            .gen_audio => impl.ggmlAssert(self.has_gen_audio, "clip.cpp", 1254, "has_gen_audio"),
        }
        model.modality = modality;

        // projector type
        var proj_type: ?[*:0]const u8 = null;
        {
            // default key
            try self.getStringPtr(names.KEY_PROJ_TYPE, &proj_type, false);

            // for models with mixed modalities
            if (proj_type == null or proj_type.?[0] == 0) {
                switch (modality) {
                    .vision => try self.getStringPtr(names.KEY_VISION_PROJ_TYPE, &proj_type, false),
                    .audio => try self.getStringPtr(names.KEY_AUDIO_PROJ_TYPE, &proj_type, false),
                    .gen_audio => try self.getStringPtr(names.KEY_GEN_AUDIO_PROJ_TYPE, &proj_type, false),
                }
            }
            if (proj_type == null) proj_type = "";

            model.proj_type = names.projectorTypeFromString(std.mem.span(proj_type.?));

            if (model.proj_type == .unknown) return throwErr("%s: unknown projector type: %s\n", .{ func, proj_type.? });

            // correct arch for multimodal models (legacy method)
            if (model.proj_type == .qwen25o) {
                model.proj_type = if (modality == .vision) .qwen25vl else .qwen2a;
            }
        }

        const is_vision = model.modality == .vision;
        const is_audio = model.modality == .audio;
        const is_gen_audio = model.modality == .gen_audio;

        // other hparams
        {
            const prefix: [*:0]const u8 = if (is_vision) "vision" else (if (is_audio) "audio" else "gen.audio");
            var kb: [256]u8 = undefined;
            try self.getU32(fmtKey(&kb, names.KEY_N_EMBD, prefix), &hparams.n_embd, true);
            try self.getU32(fmtKey(&kb, names.KEY_N_HEAD, prefix), &hparams.n_head, true);
            try self.getU32(fmtKey(&kb, names.KEY_N_EMBD_HEAD, prefix), &hparams.n_embd_head, false);
            try self.getU32(fmtKey(&kb, names.KEY_N_FF, prefix), &hparams.n_ff, true);
            try self.getU32(fmtKey(&kb, names.KEY_N_BLOCK, prefix), &hparams.n_layer, true);
            try self.getU32(fmtKey(&kb, names.KEY_PROJ_DIM, prefix), &hparams.projection_dim, true);
            try self.getF32(fmtKey(&kb, names.KEY_LAYER_NORM_EPS, prefix), &hparams.eps, true);

            // n_head_kv is optional (for GQA), default to n_head
            hparams.n_head_kv = hparams.n_head;
            try self.getU32(fmtKey(&kb, names.KEY_N_HEAD_KV, prefix), &hparams.n_head_kv, false);

            if (is_vision) {
                try self.getU32(names.KEY_IMAGE_SIZE, &hparams.image_size, true);
                try self.getU32(names.KEY_PATCH_SIZE, &hparams.patch_size, true);
                try self.getI32(names.KEY_MINICPMV_VERSION, &hparams.minicpmv_version, false); // legacy
                try self.getU32(names.KEY_MINICPMV_QUERY_NUM, &hparams.minicpmv_query_num, false);
                if (hparams.minicpmv_query_num == 0) {
                    // Fallback to hardcoded values for legacy models
                    hparams.minicpmv_query_num = switch (hparams.minicpmv_version) {
                        3, 4, 5, 6, 100045 => 64,
                        else => 96,
                    };
                }
            } else if (is_audio) {
                try self.getU32(names.KEY_A_NUM_MEL_BINS, &hparams.n_mel_bins, true);
                // some hparams are unused, but still need to set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;
            } else if (is_gen_audio) {
                // these are unused, but still need to be set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;
                try self.getString(names.KEY_GEN_AUDIO_VARIANT, &hparams.gen_model_variant, false);
            } else {
                impl.ggmlAssert(false, "clip.cpp", 1345, "false && \"unknown modality\"");
            }

            // for pinpoints, we need to convert it into a list of resolution candidates
            {
                var pinpoints: std.ArrayList(i32) = .empty;
                defer pinpoints.deinit(gpa);
                try self.getArrInt(names.KEY_IMAGE_GRID_PINPOINTS, &pinpoints, false);
                if (pinpoints.items.len % 2 != 0) {
                    return throwErr("%s: image_grid_pinpoints must have an even number of elements, got %zu\n", .{ func, pinpoints.items.len });
                }
                var i: usize = 0;
                while (i < pinpoints.items.len) : (i += 2) {
                    try hparams.image_res_candidates.append(gpa, .{ .width = pinpoints.items[i], .height = pinpoints.items[i + 1] });
                }
            }

            // default warmup value
            hparams.warmup_image_size = hparams.image_size;

            {
                var use_gelu = false;
                var use_silu = false;
                try self.getBool(names.KEY_USE_GELU, &use_gelu, false);
                try self.getBool(names.KEY_USE_SILU, &use_silu, false);
                if (use_gelu and use_silu) return throwErr("%s: both use_gelu and use_silu are set to true\n", .{func});
                if (use_gelu) {
                    hparams.ffn_op = .gelu;
                    log_ffn_op = "gelu";
                } else if (use_silu) {
                    hparams.ffn_op = .silu;
                    log_ffn_op = "silu";
                } else {
                    hparams.ffn_op = .gelu_quick;
                    log_ffn_op = "gelu_quick";
                }
            }

            {
                var mm_patch_merge_type: ?[*:0]const u8 = null;
                try self.getStringPtr(names.KEY_MM_PATCH_MERGE_TYPE, &mm_patch_merge_type, false);
                if (mm_patch_merge_type != null and std.mem.eql(u8, std.mem.span(mm_patch_merge_type.?), "spatial_unpad")) {
                    hparams.mm_patch_merge_type = .spatial_unpad;
                }
            }

            if (is_vision) {
                var image_mean: std.ArrayList(f32) = .empty;
                defer image_mean.deinit(gpa);
                var image_std: std.ArrayList(f32) = .empty;
                defer image_std.deinit(gpa);
                try self.getArrF32(names.KEY_IMAGE_MEAN, &image_mean, false);
                try self.getArrF32(names.KEY_IMAGE_STD, &image_std, false);
                if (image_mean.items.len < 3 or image_std.items.len < 3) {
                    return throwErr("%s: image_mean/image_std arrays must have at least 3 elements, got %zu and %zu\n", .{ func, image_mean.items.len, image_std.items.len });
                }
                for (0..3) |i| {
                    hparams.image_mean[i] = image_mean.items[i];
                    hparams.image_std[i] = image_std.items[i];
                }
            }

            // Load the vision/audio feature layer indices if they are explicitly provided
            try self.getArrInt(fmtKey(&kb, names.KEY_FEATURE_LAYERS, prefix), &hparams.feature_layers, false);

            // model-specific params
            try arms_hparams.load(self, model, .{
                .proj_type = proj_type.?,
                .is_vision = is_vision,
                .is_audio = is_audio,
                .is_gen_audio = is_gen_audio,
                .log_ffn_op = &log_ffn_op,
            });

            // sanity check
            {
                if (hparams.image_size < 0) {
                    // note: some models having hparams.image_size == 0, which means the image size is dynamic
                    return throwErr("%s: image_size (%d) cannot be negative\n", .{ func, hparams.image_size });
                }
                if (hparams.image_size > 8192) {
                    // cap prevents int32 overflow in n_patches = (image_size/patch_size)^2
                    return throwErr("%s: image_size (%d) is too large (max 8192)\n", .{ func, hparams.image_size });
                }
                if (hparams.patch_size <= 0 or hparams.patch_size >= 65536) {
                    return throwErr("%s: patch_size (%d) must be positive and less than 65536\n", .{ func, hparams.patch_size });
                }
                if (hparams.n_embd <= 0) return throwErr("%s: n_embd (%d) must be greater than 0\n", .{ func, hparams.n_embd });
                if (hparams.image_max_pixels < hparams.image_min_pixels) {
                    return throwErr("%s: image_max_pixels (%d) is less than image_min_pixels (%d)\n", .{ func, hparams.image_max_pixels, hparams.image_min_pixels });
                }
                if (hparams.n_merge <= 0 or hparams.n_merge >= 65536) {
                    return throwErr("%s: n_merge (%d) must be greater than 0 and less than 65536\n", .{ func, hparams.n_merge });
                }
                if (hparams.attn_window_size > 4096) {
                    return throwErr("%s: attn_window_size (%d) is too large (max 4096)\n", .{ func, hparams.attn_window_size });
                }
            }

            impl.logInf("%s: projector:          %s\n", .{ func, proj_type.? });
            impl.logInf("%s: n_embd:             %d\n", .{ func, hparams.n_embd });
            impl.logInf("%s: n_head:             %d\n", .{ func, hparams.n_head });
            impl.logInf("%s: n_ff:               %d\n", .{ func, hparams.n_ff });
            impl.logInf("%s: n_layer:            %d\n", .{ func, hparams.n_layer });
            impl.logInf("%s: ffn_op:             %s\n", .{ func, log_ffn_op });
            impl.logInf("%s: projection_dim:     %d\n", .{ func, hparams.projection_dim });
            if (is_vision) {
                impl.logInf("\n--- vision hparams ---\n", .{});
                impl.logInf("%s: image_size:         %d\n", .{ func, hparams.image_size });
                impl.logInf("%s: patch_size:         %d\n", .{ func, hparams.patch_size });
                impl.logInf("%s: has_llava_proj:     %d\n", .{ func, @as(c_int, @intFromBool(hparams.has_llava_projector)) });
                impl.logInf("%s: minicpmv_version:   %d\n", .{ func, hparams.minicpmv_version });
                impl.logInf("%s: n_merge:            %d\n", .{ func, hparams.n_merge });
                impl.logInf("%s: n_wa_pattern: %d\n", .{ func, hparams.n_wa_pattern });
                if (hparams.wa_layer_indexes.count() != 0) {
                    impl.logInf("%s: wa_layer_indexes:  ", .{func});
                    for (try arms_hparams.waLayerIndexesInSetOrder(hparams)) |layer| impl.logInf("%d ", .{layer});
                    impl.logInf("\n", .{});
                }
                if (hparams.image_min_pixels > 0) {
                    impl.logInf("%s: image_min_pixels:   %d%s\n", .{ func, hparams.image_min_pixels, @as([*:0]const u8, if (hparams.custom_image_min_tokens > 0) " (custom value)" else "") });
                }
                if (hparams.image_max_pixels > 0) {
                    impl.logInf("%s: image_max_pixels:   %d%s\n", .{ func, hparams.image_max_pixels, @as([*:0]const u8, if (hparams.custom_image_max_tokens > 0) " (custom value)" else "") });
                }
                if (hparams.preproc_max_tiles > 0) {
                    impl.logInf("%s: preproc_tiles:      %d - %d\n", .{ func, hparams.preproc_min_tiles, hparams.preproc_max_tiles });
                }
            } else if (is_audio) {
                impl.logInf("\n--- audio hparams ---\n", .{});
                impl.logInf("%s: n_mel_bins:         %d\n", .{ func, hparams.n_mel_bins });
                impl.logInf("%s: proj_stack_factor:  %d\n", .{ func, hparams.proj_stack_factor });
                impl.logInf("%s: audio_chunk_len:    %d\n", .{ func, hparams.audio_chunk_len });
                impl.logInf("%s: audio_sample_rate:  %d\n", .{ func, hparams.audio_sample_rate });
                impl.logInf("%s: audio_n_fft:        %d\n", .{ func, hparams.audio_n_fft });
                impl.logInf("%s: audio_window_len:   %d\n", .{ func, hparams.audio_window_len });
                impl.logInf("%s: audio_hop_len:      %d\n", .{ func, hparams.audio_hop_len });

                // GEMMA4UA and pocket-tts are encoder-free: no FFT/filterbank
                const fft_based = model.proj_type != .gemma4ua and model.proj_type != .pockettts_spkenc;

                // Validate audio hparams loaded from GGUF metadata
                if (hparams.n_mel_bins <= 0 or (fft_based and hparams.n_mel_bins > 256)) {
                    return throwErr("%s: n_mel_bins (%d) must be in range [1, 256]\n", .{ func, hparams.n_mel_bins });
                }
                if (fft_based and (hparams.audio_sample_rate <= 0 or hparams.audio_n_fft <= 0 or hparams.audio_hop_len <= 0 or hparams.audio_window_len <= 0)) {
                    return throwErr("%s: audio hparams invalid: sample_rate=%d n_fft=%d window_len=%d hop_len=%d\n", .{ func, hparams.audio_sample_rate, hparams.audio_n_fft, hparams.audio_window_len, hparams.audio_hop_len });
                }
            }
            impl.logInf("\n", .{});
            impl.logInf("%s: model size:         %.2f MiB\n", .{ func, @as(f64, @floatFromInt(self.model_size)) / 1024.0 / 1024.0 });
            impl.logInf("%s: metadata size:      %.2f MiB\n", .{ func, @as(f64, @floatFromInt(c.ggml_get_mem_size(self.ctx_meta))) / 1024.0 / 1024.0 });
        }
    }

    /// Ports `load_tensors` (mtmd/clip.cpp:2099 @b29c606e2), a member of `clip_model_loader`;
    /// the model-specific switch is `arms_tensors.load`.
    pub fn loadTensors(self: *Loader, ctx_clip: *Ctx) Error!void {
        const func = "load_tensors";
        const model = &ctx_clip.model;
        const hparams = &model.hparams;

        var tl: TensorLoader = .{ .loader = self, .ctx_clip = ctx_clip };
        defer tl.deinit();

        tl.fin = std.c.fopen(self.fname.ptr, "rb");
        if (tl.fin == null) return throwErr("%s: failed to open %s\n", .{ func, self.fname.ptr });

        const prefix: [*:0]const u8 = if (model.modality == .audio) "a" else if (model.modality == .gen_audio) "a.gen.code" else "v";
        tl.prefix = prefix;

        // get offsets
        {
            var i: i64 = 0;
            while (i < c.gguf_get_n_tensors(self.ctx_gguf)) : (i += 1) {
                const tname = std.mem.span(c.gguf_get_tensor_name(self.ctx_gguf, i));
                try tl.tensor_offset.put(gpa, tname, c.gguf_get_data_offset(self.ctx_gguf) + c.gguf_get_tensor_offset(self.ctx_gguf, i));
            }
        }

        // create data context
        const params: c.ggml_init_params = .{
            .mem_size = @as(usize, @intCast(c.gguf_get_n_tensors(self.ctx_gguf) + 1)) * c.ggml_tensor_overhead(),
            .mem_buffer = null,
            .no_alloc = true,
        };
        ctx_clip.ctx_data = c.ggml_init(params);
        if (ctx_clip.ctx_data == null) return throwErr("%s: failed to init ggml context\n", .{func});

        var nb: [256]u8 = undefined;
        model.class_embedding = try tl.getTensor(names.TN_CLASS_EMBD, false);

        model.pre_ln_w = try tl.getTensor(fmtKey2(&nb, names.TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = try tl.getTensor(fmtKey2(&nb, names.TN_LN_PRE, prefix, "bias"), false);

        model.post_ln_w = try tl.getTensor(fmtKey2(&nb, names.TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = try tl.getTensor(fmtKey2(&nb, names.TN_LN_POST, prefix, "bias"), false);

        model.patch_bias = try tl.getTensor(names.TN_PATCH_BIAS, false);
        model.patch_embeddings_0 = try tl.getTensor(names.TN_PATCH_EMBD, false);
        model.patch_embeddings_1 = try tl.getTensor(names.TN_PATCH_EMBD_1, false);

        model.norm_embd_w = try tl.getTensor(fmtKey(&nb, names.TN_NORM_EMBD, "weight"), false);
        model.norm_embd_b = try tl.getTensor(fmtKey(&nb, names.TN_NORM_EMBD, "bias"), false);

        model.position_embeddings = try tl.getTensor(fmtKey(&nb, names.TN_POS_EMBD, prefix), false);

        const has_standard_layers = model.proj_type != .gemma3nv and model.proj_type != .qwen3tts_spkenc and model.proj_type != .pockettts_gen;

        // layers
        const n_layers_to_load: usize = if (has_standard_layers) @intCast(hparams.n_layer) else 0;
        try model.layers.resize(gpa, n_layers_to_load);
        for (model.layers.items) |*l| l.* = .{};
        for (0..n_layers_to_load) |ilu| {
            const il: c_int = @intCast(ilu);
            const layer = &model.layers.items[ilu];
            layer.k_w = try tl.getTensor(fmtL(&nb, names.TN_ATTN_K, prefix, il, "weight"), false);
            layer.q_w = try tl.getTensor(fmtL(&nb, names.TN_ATTN_Q, prefix, il, "weight"), false);
            layer.v_w = try tl.getTensor(fmtL(&nb, names.TN_ATTN_V, prefix, il, "weight"), false);
            layer.o_w = try tl.getTensor(fmtL(&nb, names.TN_ATTN_OUTPUT, prefix, il, "weight"), true);
            layer.qkv_w = try tl.getTensor(fmtL(&nb, names.TN_ATTN_QKV, prefix, il, "weight"), false);
            layer.k_norm = try tl.getTensor(fmtL(&nb, names.TN_ATTN_K_NORM, prefix, il, "weight"), false);
            layer.q_norm = try tl.getTensor(fmtL(&nb, names.TN_ATTN_Q_NORM, prefix, il, "weight"), false);
            layer.ln_1_w = try tl.getTensor(fmtL(&nb, names.TN_LN_1, prefix, il, "weight"), false);
            layer.ln_2_w = try tl.getTensor(fmtL(&nb, names.TN_LN_2, prefix, il, "weight"), false);
            layer.ls_1_w = try tl.getTensor(fmtL(&nb, names.TN_LS_1, prefix, il, "weight"), false); // no bias
            layer.ls_2_w = try tl.getTensor(fmtL(&nb, names.TN_LS_2, prefix, il, "weight"), false); // no bias
            layer.ls_out_w = try tl.getTensor(fmtL(&nb, names.TN_LS_OUT, prefix, il, "weight"), false); // no bias
            layer.attn_post_norm_w = try tl.getTensor(fmtL(&nb, names.TN_ATTN_POST_NORM, prefix, il, "weight"), false); // no bias
            layer.ff_post_norm_w = try tl.getTensor(fmtL(&nb, names.TN_FFN_POST_NORM, prefix, il, "weight"), false); // no bias

            layer.k_b = try tl.getTensor(fmtL(&nb, names.TN_ATTN_K, prefix, il, "bias"), false);
            layer.q_b = try tl.getTensor(fmtL(&nb, names.TN_ATTN_Q, prefix, il, "bias"), false);
            layer.v_b = try tl.getTensor(fmtL(&nb, names.TN_ATTN_V, prefix, il, "bias"), false);
            layer.o_b = try tl.getTensor(fmtL(&nb, names.TN_ATTN_OUTPUT, prefix, il, "bias"), false);
            layer.qkv_b = try tl.getTensor(fmtL(&nb, names.TN_ATTN_QKV, prefix, il, "bias"), false);
            layer.ln_1_b = try tl.getTensor(fmtL(&nb, names.TN_LN_1, prefix, il, "bias"), false);
            layer.ln_2_b = try tl.getTensor(fmtL(&nb, names.TN_LN_2, prefix, il, "bias"), false);

            // MoE ffn (dots3note vision pyramid blocks); replaces the dense ffn when present
            layer.ff_gate_inp_w = try tl.getTensor(fmtL(&nb, names.TN_FFN_GATE_INP, prefix, il, "weight"), false);
            layer.ff_gate_exps_w = try tl.getTensor(fmtL(&nb, names.TN_FFN_GATE_EXPS, prefix, il, "weight"), false);
            layer.ff_up_exps_w = try tl.getTensor(fmtL(&nb, names.TN_FFN_UP_EXPS, prefix, il, "weight"), false);
            layer.ff_down_exps_w = try tl.getTensor(fmtL(&nb, names.TN_FFN_DOWN_EXPS, prefix, il, "weight"), false);
            layer.ff_exp_probs_b = try tl.getTensor(fmtL(&nb, names.TN_FFN_EXP_PROBS_B, prefix, il, "weight"), false);
            const is_moe = layer.ff_gate_exps_w != null;

            // ffn
            layer.ff_up_w = try tl.getTensor(fmtL(&nb, names.TN_FFN_UP, prefix, il, "weight"), !is_moe);
            layer.ff_up_b = try tl.getTensor(fmtL(&nb, names.TN_FFN_UP, prefix, il, "bias"), false);
            layer.ff_gate_w = try tl.getTensor(fmtL(&nb, names.TN_FFN_GATE, prefix, il, "weight"), false);
            layer.ff_gate_b = try tl.getTensor(fmtL(&nb, names.TN_FFN_GATE, prefix, il, "bias"), false);
            layer.ff_down_w = try tl.getTensor(fmtL(&nb, names.TN_FFN_DOWN, prefix, il, "weight"), !is_moe);
            layer.ff_down_b = try tl.getTensor(fmtL(&nb, names.TN_FFN_DOWN, prefix, il, "bias"), false);

            // mimovl per-head attention sink bias
            layer.attn_sinks = try tl.getTensor(fmtPI(&nb, names.TN_ATTN_SINKS, prefix, il), false);

            // qwen3vl deepstack layer
            layer.deepstack_norm_w = try tl.getTensor(fmtIS(&nb, names.TN_DEEPSTACK_NORM, il, "weight"), false);
            layer.deepstack_norm_b = try tl.getTensor(fmtIS(&nb, names.TN_DEEPSTACK_NORM, il, "bias"), false);
            layer.deepstack_fc1_w = try tl.getTensor(fmtIS(&nb, names.TN_DEEPSTACK_FC1, il, "weight"), false);
            layer.deepstack_fc1_b = try tl.getTensor(fmtIS(&nb, names.TN_DEEPSTACK_FC1, il, "bias"), false);
            layer.deepstack_fc2_w = try tl.getTensor(fmtIS(&nb, names.TN_DEEPSTACK_FC2, il, "weight"), false);
            layer.deepstack_fc2_b = try tl.getTensor(fmtIS(&nb, names.TN_DEEPSTACK_FC2, il, "bias"), false);
            if (layer.hasDeepstack()) model.n_deepstack_layers += 1;

            // some models already exported with legacy (incorrect) naming which is quite messy, let's fix it here
            // note: Qwen model converted from the old surgery script has n_ff = 0, so we cannot use n_ff to check!
            const pt = model.proj_type;
            const is_ffn_swapped = (pt == .mlp or pt == .mlp_norm or pt == .ldp or pt == .ldpv2 or pt == .qwen2vl or pt == .qwen25vl or
                pt == .exaone4_5 or pt == .glm_edge or pt == .gemma3 or pt == .idefics3 or pt == .minicpmv or pt == .minicpmv4_6) and
                layer.ff_up_w != null and layer.ff_down_w != null and graph_mod.ne(layer.ff_down_w, 0) == hparams.n_embd;
            if (is_ffn_swapped) {
                // swap up and down weights
                std.mem.swap(T, &layer.ff_up_w, &layer.ff_down_w);
                // swap up and down biases
                std.mem.swap(T, &layer.ff_up_b, &layer.ff_down_b);
                if (il == 0) impl.logWrn("%s: ffn up/down are swapped\n", .{func});
            }
        }

        // model-specific tensors
        try arms_tensors.load(&tl);

        // load data
        {
            var read_buf: std.ArrayList(u8) = .empty;
            defer read_buf.deinit(gpa);

            // start loading event
            if (self.progress_callback) |cb| _ = cb(0.0, self.progress_callback_user_data);

            // compute total tensor data size for progress reporting
            var total_data_size: usize = 0;
            for (tl.tensors_to_load.items) |t| total_data_size += c.ggml_nbytes(t);

            // alloc memory and offload data
            const buft = c.ggml_backend_get_default_buffer_type(ctx_clip.backend);
            ctx_clip.buf = c.ggml_backend_alloc_ctx_tensors_from_buft(ctx_clip.ctx_data, buft);
            c.ggml_backend_buffer_set_usage(ctx_clip.buf, c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // read the weight from file
            if (!ctx_clip.no_alloc) {
                var data_loaded: usize = 0;
                for (tl.tensors_to_load.items) |t| {
                    const tname: [*:0]const u8 = @ptrCast(&t.?.name);
                    const cur = c.ggml_get_tensor(ctx_clip.ctx_data, tname);
                    impl.ggmlAssert(cur != null, "clip.cpp", 3608, "cur && \"tensor not found in ctx_data\"");
                    const off = tl.tensor_offset.get(std.mem.span(tname));
                    impl.ggmlAssert(off != null, "clip.cpp", 3610, "it_off != tensor_offset.end() && \"no offset for tensor\"");
                    if (fseeko(tl.fin, @intCast(off.?), 0) != 0) {
                        return throwErr("%s: failed to seek for tensor %s\n", .{ func, tname });
                    }
                    const num_bytes = c.ggml_nbytes(cur);
                    if (c.ggml_backend_buft_is_host(buft)) {
                        // for the CPU and Metal backend, we can read directly into the tensor
                        _ = std.c.fread(@ptrCast(cur.*.data), 1, num_bytes, tl.fin.?);
                    } else {
                        // read into a temporary buffer first, then copy to device memory
                        try read_buf.resize(gpa, num_bytes);
                        _ = std.c.fread(read_buf.items.ptr, 1, num_bytes, tl.fin.?);
                        c.ggml_backend_tensor_set(cur, read_buf.items.ptr, 0, num_bytes);
                    }
                    data_loaded += num_bytes;
                    if (self.progress_callback) |cb| {
                        if (total_data_size > 0) {
                            const progress = @as(f32, @floatFromInt(data_loaded)) / @as(f32, @floatFromInt(total_data_size));
                            if (!cb(progress, self.progress_callback_user_data)) {
                                return throwErr("%s: model loading cancelled by progress_callback\n", .{func});
                            }
                        }
                    }
                }
                impl.logDbg("%s: loaded %zu tensors from %s\n", .{ func, tl.tensors_to_load.items.len, self.fname.ptr });
            } else {
                impl.logDbg("%s: no_alloc is set, skipping tensor data loading (%zu tensors)\n", .{ func, tl.tensors_to_load.items.len });
            }
        }
    }

    /// Ports `get_dummy_batch` (mtmd/clip.cpp:3658 @b29c606e2), a member of `clip_model_loader`.
    pub fn getDummyBatch(ctx_clip: *Ctx) Error!impl.ImageF32Batch {
        const func = "get_dummy_batch";
        // create a fake batch
        const hparams = &ctx_clip.model.hparams;
        var batch: impl.ImageF32Batch = .{};
        var img: impl.ImageF32 = .{};
        if (ctx_clip.model.modality == .vision) {
            const sz = hparams.warmup_image_size;
            try img.setSize(.{ .width = sz, .height = sz }, false, false);
            impl.logInf("%s: warmup with image size = %d x %d\n", .{ func, sz, sz });
        } else {
            // GEMMA4UA uses n_mel_bins as a raw-waveform frame size (640), not a mel-bin count,
            // so the [1, 256] bound only applies to FFT-based models.
            const fft_based = ctx_clip.model.proj_type != .gemma4ua;
            if (hparams.n_mel_bins <= 0 or (fft_based and hparams.n_mel_bins > 256)) {
                img.deinit();
                return throwErr("%s: invalid n_mel_bins (%d), must be in [1, 256]\n", .{ func, hparams.n_mel_bins });
            }
            try img.setSize(.{ .width = hparams.warmup_audio_size, .height = hparams.n_mel_bins }, false, false);
            impl.logInf("%s: warmup with audio size = %d\n", .{ func, hparams.warmup_audio_size });
        }
        try batch.entries.append(gpa, img);
        return batch;
    }

    /// Ports `init_ctx` (mtmd/clip.cpp:3681 @b29c606e2), a member of `clip_model_loader`.
    pub fn initCtx(ctx_clip: *Ctx) Error!void {
        try ctx_clip.buf_compute_meta.resize(gpa, @as(usize, @intCast(ctx_clip.max_nodes)) * c.ggml_tensor_overhead() + c.ggml_graph_overhead());

        // check batching support: the C++ constructs a builder for the
        // dummy batch and asks it, without building
        var batch = try getDummyBatch(ctx_clip);
        defer batch.deinit();
        ctx_clip.support_batch = models.supportBatch(ctx_clip.projType()) orelse impl.abort("clip.cpp", 1143, "missing cgraph builder");
    }

    /// Ports `warmup` (mtmd/clip.cpp:3690 @b29c606e2), a member of
    /// `clip_model_loader`: the overload over a context.
    pub fn warmup(ctx_clip: *Ctx) Error!void {
        var batch = try getDummyBatch(ctx_clip);
        defer batch.deinit();
        try warmupBatch(ctx_clip, &batch);
    }

    const SupportInfoOp = struct {
        op: T,
        // true if the op runs on the accelerated ctx_clip.backend
        is_accel: bool = true,
    };

    const SupportInfoGraph = struct {
        // whether the clip_ctx.backend supports flash attention
        fattn: bool = true,
        fattn_op: T = null, // for debugging
        ops: std.ArrayList(SupportInfoOp) = .empty,

        fn deinit(self: *SupportInfoGraph) void {
            self.ops.deinit(gpa);
        }
    };

    /// Ports `warmup` (mtmd/clip.cpp:3695 @b29c606e2), a member of
    /// `clip_model_loader`: the overload over a context and a batch.
    pub fn warmupBatch(ctx_clip: *Ctx, batch: *const impl.ImageF32Batch) Error!void {
        const func = "warmup";
        var info: SupportInfoGraph = .{};
        defer info.deinit();

        if (ctx_clip.flash_attn_type == .auto) {
            // try to enable flash attention to see if it's supported
            ctx_clip.flash_attn_type = .enabled;
            info.deinit();
            info = try reserveComputeMeta(ctx_clip, batch);
            if (!info.fattn and info.fattn_op != null) {
                const op = info.fattn_op;
                impl.logWrn("%s: *****************************************************************\n", .{func});
                impl.logWrn("%s: WARNING: flash attention not supported by %s, memory usage will increase\n", .{ func, c.ggml_backend_name(ctx_clip.backend) });
                impl.logWrn("%s: op params: \n", .{func});
                printShape(func, " dst", op);
                printShape(func, "src0", op.?.src[0]);
                printShape(func, "src1", op.?.src[1]);
                printShape(func, "src2", op.?.src[2]);
                impl.logWrn("%s: please report this on github as an issue\n", .{func});
                impl.logWrn("%s: *****************************************************************\n", .{func});
                ctx_clip.flash_attn_type = .disabled;
                var info2 = try reserveComputeMeta(ctx_clip, batch);
                info2.deinit();
            }
        } else {
            info.deinit();
            info = try reserveComputeMeta(ctx_clip, batch);
            if (!info.fattn and ctx_clip.flash_attn_type == .enabled) {
                impl.logWrn("%s: flash attention is not supported by the current backend; falling back to CPU (performance will be degraded)\n", .{func});
            }
        }

        ctx_clip.is_allocated = true; // mark buffers as allocated

        impl.logInf("%s: flash attention is %s\n", .{ func, @as([*:0]const u8, if (ctx_clip.flash_attn_type == .enabled) "enabled" else "disabled") });

        // print ops that are not supported by the GPU backend (if there is one)
        if (ctx_clip.backend != null and ctx_clip.backend != ctx_clip.backend_cpu) {
            var unsupported_ops: std.ArrayList(SupportInfoOp) = .empty;
            defer unsupported_ops.deinit(gpa);
            for (info.ops.items) |op| {
                if (!op.is_accel) try unsupported_ops.append(gpa, op);
            }
            if (unsupported_ops.items.len != 0) {
                impl.logWrn("%s: *****************************************************************\n", .{func});
                impl.logWrn("%s: WARNING: the CLIP graph uses unsupported operators by the backend\n", .{func});
                impl.logWrn("%s:          the performance will be suboptimal                      \n", .{func});
                impl.logWrn("%s:          list of unsupported ops (backend=%s):\n", .{ func, c.ggml_backend_name(ctx_clip.backend) });
                for (unsupported_ops.items) |op| {
                    const t = op.op.?;
                    impl.logWrn("%s: %16s: type = %s, ne = [%d %d %d %d]\n", .{
                        func,                           c.ggml_op_name(t.op),           c.ggml_type_name(t.type),
                        @as(c_int, @truncate(t.ne[0])), @as(c_int, @truncate(t.ne[1])), @as(c_int, @truncate(t.ne[2])),
                        @as(c_int, @truncate(t.ne[3])),
                    });
                }
                impl.logWrn("%s: flash attention is %s\n", .{ func, @as([*:0]const u8, if (ctx_clip.flash_attn_type == .enabled) "enabled" else "disabled") });
                impl.logWrn("%s: please report this on github as an issue\n", .{func});
                impl.logWrn("%s: ref: https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118\n", .{func});
                impl.logWrn("%s: *****************************************************************\n", .{func});
            }
        }
    }

    /// The `print_shape` lambda of `warmup` (mtmd/clip.cpp:3695 @b29c606e2): `%d`
    /// over `int64_t` and `size_t`, which the C++ passes as 64-bit values
    /// and `printf` reads as their low 32 bits on this ABI.
    fn printShape(func: [*:0]const u8, name: [*:0]const u8, t: T) void {
        const x = t.?;
        impl.logWrn("%s:   %s: type = %s, ne = [%d %d %d %d], nb = [%d %d %d %d]\n", .{
            func,                                               name,                                               c.ggml_type_name(x.type),
            @as(c_int, @truncate(x.ne[0])),                     @as(c_int, @truncate(x.ne[1])),                     @as(c_int, @truncate(x.ne[2])),
            @as(c_int, @truncate(x.ne[3])),                     @as(c_int, @truncate(@as(i64, @bitCast(x.nb[0])))), @as(c_int, @truncate(@as(i64, @bitCast(x.nb[1])))),
            @as(c_int, @truncate(@as(i64, @bitCast(x.nb[2])))), @as(c_int, @truncate(@as(i64, @bitCast(x.nb[3])))),
        });
    }

    /// Ports `reserve_compute_meta` (mtmd/clip.cpp:3763
    /// @b29c606e2), a member of `clip_model_loader`: initialize the backend buffers, without allocating.
    fn reserveComputeMeta(ctx_clip: *Ctx, batch: *const impl.ImageF32Batch) Error!SupportInfoGraph {
        const func = "reserve_compute_meta";
        var built = try models.buildGraph(ctx_clip, batch, null);
        defer built.deinit();
        const gf = built.gf;
        _ = c.ggml_backend_sched_reserve(ctx_clip.sched, gf);

        ctx_clip.mem_compute.clear();
        for (ctx_clip.backend_ptrs.items, 0..) |backend, i| {
            const buft = ctx_clip.backend_buft.items[i];
            const size = c.ggml_backend_sched_get_buffer_size(ctx_clip.sched, backend);
            if (size > 1) {
                impl.logInf("%s: %10s compute buffer size = %8.2f MiB\n", .{ func, c.ggml_backend_buft_name(buft), @as(f64, @floatFromInt(size)) / 1024.0 / 1024.0 });
            }
            try ctx_clip.mem_compute.add(c.ggml_backend_get_device(backend), size);
        }

        const n_splits = c.ggml_backend_sched_get_n_splits(ctx_clip.sched);
        const n_nodes = c.ggml_graph_n_nodes(gf);

        impl.logInf("%s: graph splits = %d, nodes = %d\n", .{ func, n_splits, n_nodes });

        var res: SupportInfoGraph = .{};

        // check op support
        var i: c_int = 0;
        while (i < c.ggml_graph_n_nodes(gf)) : (i += 1) {
            const node = c.ggml_graph_node(gf, i);
            try res.ops.append(gpa, .{ .op = node, .is_accel = true });
            if (!c.ggml_backend_supports_op(ctx_clip.backend, node)) {
                res.ops.items[res.ops.items.len - 1].is_accel = false;
                if (node.*.op == c.GGML_OP_FLASH_ATTN_EXT) {
                    res.fattn = false;
                    res.fattn_op = node;
                }
            }
        }
        return res;
    }
};

extern fn fseeko(f: ?*std.c.FILE, off: i64, whence: c_int) c_int;

/// `load_tensors`'s locals and lambdas, handed to `arms_tensors.load`.
pub const TensorLoader = struct {
    loader: *Loader,
    ctx_clip: *Ctx,
    prefix: [*:0]const u8 = "v",
    fin: ?*std.c.FILE = null,
    tensor_offset: std.StringHashMapUnmanaged(usize) = .empty,
    tensors_to_load: std.ArrayList(T) = .empty,
    loaded_tensor_names: std.StringHashMapUnmanaged(void) = .empty,

    fn deinit(self: *TensorLoader) void {
        if (self.fin) |f| _ = std.c.fclose(f);
        self.tensor_offset.deinit(gpa);
        self.tensors_to_load.deinit(gpa);
        var it = self.loaded_tensor_names.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        self.loaded_tensor_names.deinit(gpa);
    }

    /// The `get_tensor` lambda (mtmd/clip.cpp:2134 @b29c606e2).
    pub fn getTensor(self: *TensorLoader, name: [*:0]const u8, required: bool) Error!T {
        const func = "load_tensors";
        const key = std.mem.span(name);
        // Each tensor should only be loaded once; duplicates indicate a bug
        if (self.loaded_tensor_names.contains(key)) return throwErr("%s: tensor already loaded: %s\n", .{ func, name });
        var cur: T = c.ggml_get_tensor(self.loader.ctx_meta, name);
        if (cur == null and required) return throwErr("%s: unable to find tensor %s\n", .{ func, name });
        if (cur != null) {
            try self.tensors_to_load.append(gpa, cur);
            const data_tensor = c.ggml_dup_tensor(self.ctx_clip.ctx_data, cur);
            _ = c.ggml_set_name(data_tensor, @ptrCast(&cur.?.name));
            try self.loaded_tensor_names.put(gpa, try gpa.dupe(u8, key), {});
            cur = data_tensor;
            // add to weight memory counter
            try self.ctx_clip.mem_usage.add(c.ggml_backend_get_device(self.ctx_clip.backend), c.ggml_nbytes(cur));
        }
        return cur;
    }

    /// The `get_vector` lambda (mtmd/clip.cpp:2180 @b29c606e2). The caller owns
    /// the list.
    pub fn getVector(self: *TensorLoader, name: [*:0]const u8) Error!std.ArrayList(f32) {
        const func = "load_tensors";
        var result: std.ArrayList(f32) = .empty;
        const off = self.tensor_offset.get(std.mem.span(name)) orelse return result;

        const idx = c.gguf_find_tensor(self.loader.ctx_gguf, name);
        if (idx < 0) return throwErr("%s: failed to find tensor %s\n", .{ func, name });

        const t = c.gguf_get_tensor_type(self.loader.ctx_gguf, idx);
        if (t != c.GGML_TYPE_F32) return throwErr("%s: %s must be %s, was %s\n", .{ func, name, c.ggml_type_name(c.GGML_TYPE_F32), c.ggml_type_name(t) });

        const n_bytes = c.gguf_get_tensor_size(self.loader.ctx_gguf, idx);
        if (n_bytes == 0) return throwErr("%s: tensor %s is empty\n", .{ func, name });

        const n_elems = n_bytes / @sizeOf(f32);
        try result.resize(gpa, n_elems);
        _ = fseeko(self.fin, @intCast(off), 0);
        _ = std.c.fread(@ptrCast(result.items.ptr), 1, n_bytes, self.fin.?);
        return result;
    }

    /// The `get_scalar` lambda (mtmd/clip.cpp:2209 @b29c606e2).
    pub fn getScalar(self: *TensorLoader, name: [*:0]const u8, default_val: f32) Error!f32 {
        var v = try self.getVector(name);
        defer v.deinit(gpa);
        if (v.items.len == 0) return default_val;
        if (v.items.len != 1) return throwErr("%s: expected scalar tensor '%s' but got %d elements\n", .{ "load_tensors", name, @as(c_int, @intCast(v.items.len)) });
        return v.items[0];
    }

    /// The `load_seanet` lambda (mtmd/clip.cpp:2156 @b29c606e2): pocket-tts's
    /// encoder and decoder share a layout under different prefixes.
    pub fn loadSeanet(self: *TensorLoader, seanet: *model_mod.Seanet, is_decoder: bool) Error!void {
        const hparams = &self.ctx_clip.model.hparams;
        const conv_in: [*:0]const u8 = if (is_decoder) names.TN_A_GEN_WAV_SEANET_CONV_IN else names.TN_A_SEANET_CONV_IN;
        const conv_out: [*:0]const u8 = if (is_decoder) names.TN_A_GEN_WAV_SEANET_CONV_OUT else names.TN_A_SEANET_CONV_OUT;
        const res1: [*:0]const u8 = if (is_decoder) names.TN_A_GEN_WAV_SEANET_RES_CONV1 else names.TN_A_SEANET_RES_CONV1;
        const res2: [*:0]const u8 = if (is_decoder) names.TN_A_GEN_WAV_SEANET_RES_CONV2 else names.TN_A_SEANET_RES_CONV2;
        const scale: [*:0]const u8 = if (is_decoder) names.TN_A_GEN_WAV_SEANET_SCALE_CONV else names.TN_A_SEANET_SCALE_CONV;

        var nb: [256]u8 = undefined;
        seanet.conv_in_w = try self.getTensor(fmtRt(&nb, conv_in, .{@as([*:0]const u8, "weight")}), true);
        seanet.conv_in_b = try self.getTensor(fmtRt(&nb, conv_in, .{@as([*:0]const u8, "bias")}), true);
        seanet.conv_out_w = try self.getTensor(fmtRt(&nb, conv_out, .{@as([*:0]const u8, "weight")}), true);
        seanet.conv_out_b = try self.getTensor(fmtRt(&nb, conv_out, .{@as([*:0]const u8, "bias")}), true);

        try seanet.stages.resize(gpa, @intCast(hparams.seanet_n_stage));
        for (seanet.stages.items, 0..) |*stage, iu| {
            const i: c_int = @intCast(iu);
            stage.res_conv1_w = try self.getTensor(fmtRt(&nb, res1, .{ i, @as([*:0]const u8, "weight") }), true);
            stage.res_conv1_b = try self.getTensor(fmtRt(&nb, res1, .{ i, @as([*:0]const u8, "bias") }), true);
            stage.res_conv2_w = try self.getTensor(fmtRt(&nb, res2, .{ i, @as([*:0]const u8, "weight") }), true);
            stage.res_conv2_b = try self.getTensor(fmtRt(&nb, res2, .{ i, @as([*:0]const u8, "bias") }), true);
            stage.scale_conv_w = try self.getTensor(fmtRt(&nb, scale, .{ i, @as([*:0]const u8, "weight") }), true);
            stage.scale_conv_b = try self.getTensor(fmtRt(&nb, scale, .{ i, @as([*:0]const u8, "bias") }), true);
        }
    }
};

// `string_format` into a caller's buffer, for the key and tensor names: one
// helper per argument shape the C++ passes.

/// `string_format(fmt, s)`.
pub fn fmtKey(buf: *[256]u8, comptime fmt: [*:0]const u8, s: [*:0]const u8) [*:0]const u8 {
    _ = snprintf(buf, buf.len, fmt, s);
    return @ptrCast(buf);
}

/// `string_format(fmt, s1, s2)`.
pub fn fmtKey2(buf: *[256]u8, comptime fmt: [*:0]const u8, s1: [*:0]const u8, s2: [*:0]const u8) [*:0]const u8 {
    _ = snprintf(buf, buf.len, fmt, s1, s2);
    return @ptrCast(buf);
}

/// `string_format(fmt, prefix, il, suffix)`: a layer tensor.
pub fn fmtL(buf: *[256]u8, comptime fmt: [*:0]const u8, prefix: [*:0]const u8, il: c_int, suffix: [*:0]const u8) [*:0]const u8 {
    _ = snprintf(buf, buf.len, fmt, prefix, il, suffix);
    return @ptrCast(buf);
}

/// `string_format(fmt, prefix, il)`.
pub fn fmtPI(buf: *[256]u8, comptime fmt: [*:0]const u8, prefix: [*:0]const u8, il: c_int) [*:0]const u8 {
    _ = snprintf(buf, buf.len, fmt, prefix, il);
    return @ptrCast(buf);
}

/// `string_format(fmt, il, suffix)`.
pub fn fmtIS(buf: *[256]u8, comptime fmt: [*:0]const u8, il: c_int, suffix: [*:0]const u8) [*:0]const u8 {
    _ = snprintf(buf, buf.len, fmt, il, suffix);
    return @ptrCast(buf);
}

/// `string_format` with a format known only at run time.
pub fn fmtRt(buf: *[256]u8, fmt: [*:0]const u8, args: anytype) [*:0]const u8 {
    _ = @call(.auto, snprintf, .{ buf, buf.len, fmt } ++ args);
    return @ptrCast(buf);
}

// -----------------------------------------------------------------------------
// clip_init and the free functions

/// Ports `clip_init` (mtmd/clip.cpp:3957 @b29c606e2).
pub fn init(fname: [:0]const u8, ctx_params: ContextParams) InitResult {
    var res: InitResult = .{};
    initInner(fname, &ctx_params, &res) catch |e| {
        const msg: [:0]const u8 = if (e == error.OutOfMemory) "std::bad_alloc" else what();
        impl.logErr("%s: failed to load model '%s': %s\n", .{ "clip_init", fname.ptr, msg.ptr });
        if (res.ctx_v) |x| x.destroy();
        if (res.ctx_a) |x| x.destroy();
        if (res.ctx_gen_a) |x| x.destroy();
        return .{};
    };
    return res;
}

fn initInner(fname: [:0]const u8, ctx_params: *const ContextParams, res: *InitResult) Error!void {
    var loader = try Loader.init(fname, false, ctx_params.progress_callback, ctx_params.progress_callback_user_data);
    defer loader.deinit();
    var skip_audio = false;

    if (loader.has_vision) {
        res.ctx_v = try Ctx.create(ctx_params);
        try loader.loadHparams(&res.ctx_v.?.model, .vision);
        try loader.loadTensors(res.ctx_v.?);
        try Loader.initCtx(res.ctx_v.?);
        if (ctx_params.warmup) try Loader.warmup(res.ctx_v.?);

        // TODO: we don't support audio for Gemma 3N, but GGUF contains audio tensors
        skip_audio = res.ctx_v.?.model.proj_type == .gemma3nv;
    }

    if (loader.has_audio and !skip_audio) {
        res.ctx_a = try Ctx.create(ctx_params);
        try loader.loadHparams(&res.ctx_a.?.model, .audio);
        try loader.loadTensors(res.ctx_a.?);
        try Loader.initCtx(res.ctx_a.?);
        if (ctx_params.warmup) try Loader.warmup(res.ctx_a.?);
    }

    if (loader.has_gen_audio) {
        res.ctx_gen_a = try Ctx.create(ctx_params);
        try loader.loadHparams(&res.ctx_gen_a.?.model, .gen_audio);
        try loader.loadTensors(res.ctx_gen_a.?);
        // TODO: fix warmup
        try res.ctx_gen_a.?.buf_compute_meta.resize(gpa, @as(usize, @intCast(res.ctx_gen_a.?.max_nodes)) * c.ggml_tensor_overhead() + c.ggml_graph_overhead());
    }
}

/// Ports `clip_get_cap` (mtmd/clip.cpp:4014 @b29c606e2). A load failure is the
/// C++'s escaping exception, which `mtmd_get_cap_from_file` catches.
pub fn getCap(fname: [:0]const u8) Error!Cap {
    var loader = try Loader.init(fname, true, null, null);
    defer loader.deinit();
    return .{ .has_vision = loader.has_vision, .has_audio = loader.has_audio };
}

/// Ports `clip_free` (mtmd/clip.cpp:4022 @b29c606e2).
pub fn free(ctx: ?*Ctx) void {
    if (ctx) |x| x.destroy();
}

/// Ports `clip_patch_merge_type` (mtmd/clip.cpp:4029 @b29c606e2).
pub fn patchMergeType(ctx: *const Ctx) [*:0]const u8 {
    return if (ctx.model.hparams.mm_patch_merge_type == .spatial_unpad) "spatial_unpad" else "flat";
}

/// Ports `clip_n_output_tokens_x` (mtmd/clip.cpp:4033 @b29c606e2): for M-RoPE,
/// the token positions in X; otherwise every token.
pub fn nOutputTokensX(ctx: *const Ctx, img: *const impl.ImageF32) c_int {
    const params = &ctx.model.hparams;
    const n_total = nOutputTokens(ctx, img);
    switch (ctx.projType()) {
        .qwen2vl, .qwen25vl, .qwen3vl, .exaone4_5, .mimovl, .glm4v, .paddleocr, .hunyuanvl, .youtuvl, .muse_glimmer => return @divTrunc(@divTrunc(img.nx(), params.patch_size), 2),
        .step3vl => return @divTrunc(img.nx(), params.patch_size * params.n_merge),
        .deepseekocr, .deepseekocr2 => return @divTrunc(@divTrunc(img.nx(), params.patch_size), 4),
        else => {},
    }
    return n_total;
}

/// Ports `clip_n_output_tokens_y` (mtmd/clip.cpp:4060 @b29c606e2).
pub fn nOutputTokensY(ctx: *const Ctx, img: *const impl.ImageF32) c_int {
    const params = &ctx.model.hparams;
    switch (ctx.projType()) {
        .qwen2vl, .qwen25vl, .qwen3vl, .exaone4_5, .mimovl, .glm4v, .paddleocr, .hunyuanvl, .youtuvl, .muse_glimmer => return @divTrunc(@divTrunc(img.ny(), params.patch_size), 2),
        .step3vl => return @divTrunc(img.ny(), params.patch_size * params.n_merge),
        else => {},
    }
    return 1;
}

/// Ports `clip_n_output_tokens` (mtmd/clip.cpp:4083 @b29c606e2): the
/// per-projector body is `arms_inputs.nOutputTokens`.
pub fn nOutputTokens(ctx: *const Ctx, img: *const impl.ImageF32) c_int {
    return arms_inputs.nOutputTokens(ctx, img);
}

/// Ports `clip_image_encode` (mtmd/clip.cpp:4390 @b29c606e2).
pub fn imageEncode(ctx: *Ctx, n_threads: c_int, img: *const impl.ImageF32, out_vec: *std.ArrayList(f32)) Error!bool {
    var imgs: impl.ImageF32Batch = .{};
    defer imgs.deinit();
    try imgs.entries.append(gpa, try img.clone());
    return imageBatchEncode(ctx, n_threads, &imgs, out_vec);
}

/// Ports `clip_image_batch_encode` (mtmd/clip.cpp:4398 @b29c606e2).
pub fn imageBatchEncode(ctx: *Ctx, n_threads: c_int, imgs: *const impl.ImageF32Batch, out_batch_embd: *std.ArrayList(f32)) Error!bool {
    var params: EncodeParams = .{ .imgs = imgs, .n_threads = n_threads, .out_embd = out_batch_embd };
    return encode(ctx, &params);
}

/// What the input arms of `clip_encode` read and the helpers they call
/// (mtmd/clip.cpp:4416 @b29c606e2).
pub const EncodeState = struct {
    ctx: *Ctx,
    params: *EncodeParams,
    imgs: *const impl.ImageF32Batch,
    gf: ?*c.ggml_cgraph,

    image_size_width: c_int,
    image_size_height: c_int,
    patch_size: c_int,
    num_patches: c_int,
    n_pos: c_int,
    pos_w: c_int,
    pos_h: c_int,

    /// The `get_inp_tensor` lambda (mtmd/clip.cpp:4455 @b29c606e2).
    pub fn getInpTensor(self: *const EncodeState, name: [*:0]const u8) T {
        const inp = c.ggml_graph_get_tensor(self.gf, name);
        if (inp == null) _ = c.ggml_abort("clip.cpp", 4458, "Failed to get tensor %s", name);
        if (inp.*.flags & c.GGML_TENSOR_FLAG_INPUT == 0) _ = c.ggml_abort("clip.cpp", 4461, "Tensor %s is not an input tensor", name);
        return inp;
    }

    /// The `set_input_f32` lambda (mtmd/clip.cpp:4466 @b29c606e2).
    pub fn setInputF32(self: *const EncodeState, name: [*:0]const u8, values: []const f32) void {
        const cur = self.getInpTensor(name);
        impl.ggmlAssert(cur.?.type == c.GGML_TYPE_F32, "clip.cpp", 4468, "cur->type == GGML_TYPE_F32");
        impl.ggmlAssert(c.ggml_nelements(cur) == @as(i64, @intCast(values.len)), "clip.cpp", 4469, "ggml_nelements(cur) == (int64_t)values.size()");
        c.ggml_backend_tensor_set(cur, values.ptr, 0, c.ggml_nbytes(cur));
    }

    /// The `set_input_i32` lambda (mtmd/clip.cpp:4473 @b29c606e2).
    pub fn setInputI32(self: *const EncodeState, name: [*:0]const u8, values: []const i32) void {
        const cur = self.getInpTensor(name);
        impl.ggmlAssert(cur.?.type == c.GGML_TYPE_I32, "clip.cpp", 4475, "cur->type == GGML_TYPE_I32");
        impl.ggmlAssert(c.ggml_nelements(cur) == @as(i64, @intCast(values.len)), "clip.cpp", 4476, "ggml_nelements(cur) == (int64_t)values.size()");
        c.ggml_backend_tensor_set(cur, values.ptr, 0, c.ggml_nbytes(cur));
    }

    /// The `set_gen_state_in` lambda (mtmd/clip.cpp:4481 @b29c606e2): upload the
    /// decoder state from the previous call, or zero-fill on a cold start.
    pub fn setGenStateIn(self: *const EncodeState) Error!void {
        var offset: usize = 0;
        const slots = try models.genStateSlots(&self.ctx.model.hparams, &self.ctx.model);
        defer models.freeSlots(slots);
        for (slots) |slot| {
            var nb: [256]u8 = undefined;
            _ = snprintf(&nb, nb.len, "state_in_%s", slot.name.ptr);
            const t = self.getInpTensor(@ptrCast(&nb));
            const n = c.ggml_nbytes(t);
            if (self.params.state_in != null and self.params.state_in.?.items.len >= offset + n) {
                c.ggml_backend_tensor_set(t, self.params.state_in.?.items.ptr + offset, 0, n);
            } else {
                const zeros = try gpa.alloc(u8, n);
                defer gpa.free(zeros);
                @memset(zeros, 0);
                c.ggml_backend_tensor_set(t, zeros.ptr, 0, n);
            }
            offset += n;
        }
    }

    /// The `set_pockettts_tfm_inputs` lambda (mtmd/clip.cpp:4498 @b29c606e2): rope
    /// positions and the causal sliding-window mask of the mimi transformers.
    pub fn setPocketttsTfmInputs(self: *const EncodeState) Error!void {
        const hparams = &self.ctx.model.hparams;
        const n_pos = c.ggml_nelements(self.getInpTensor("inp_pos"));
        impl.ggmlAssert(n_pos > 0, "clip.cpp", 4500, "n_pos > 0");
        const positions = try gpa.alloc(i32, @intCast(n_pos));
        defer gpa.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(i);
        self.setInputI32("inp_pos", positions);

        // the preprocessor truncates the waveform to keep this mask bounded
        const max_pos: i64 = @divTrunc(@as(i64, model_mod.Hparams.pockettts_max_spk_seconds) * hparams.audio_sample_rate, 120);
        impl.ggmlAssert(n_pos <= max_pos, "clip.cpp", 4509, "n_pos <= max_pos && \"pocket-tts speaker reference too long for a dense mask\"");

        const context: i64 = hparams.mimi_tfm_context;
        const mask = try gpa.alloc(f32, @intCast(n_pos * n_pos));
        defer gpa.free(mask);
        @memset(mask, -std.math.inf(f32));
        var q: i64 = 0;
        while (q < n_pos) : (q += 1) {
            var k: i64 = 0;
            while (k < n_pos) : (k += 1) {
                const delta = q - k;
                if (delta >= 0 and delta < context) mask[@intCast(q * n_pos + k)] = 0.0;
            }
        }
        self.setInputF32("kq_mask", mask);
    }
};

/// Ports `clip_encode` (mtmd/clip.cpp:4416 @b29c606e2); the per-projector input
/// switch is `arms_inputs.setInputs`.
pub fn encode(ctx: *Ctx, params: *EncodeParams) Error!bool {
    const func = "clip_encode";
    const imgs = params.imgs.?;
    const n_batch_cur: c_int = @intCast(imgs.entries.items.len);

    // [QWEN_VIDEO] for video models, the batch dimension is used as temporal dimension for merged frames
    if (!ctx.support_batch and n_batch_cur > modelNTemporalMerge(ctx)) {
        impl.logErr("%s: batch size %d exceeds maximum supported batch/temporal-merge size %d\n", .{ func, n_batch_cur, modelNTemporalMerge(ctx) });
        return false;
    }

    // if buffers are not allocated, we need to do a warmup run to allocate them
    if (!ctx.is_allocated) try Loader.warmupBatch(ctx, imgs);

    if (params.seed != ctx.rng_seed) {
        ctx.rng_seed = params.seed;
        ctx.rng.seed(if (params.seed == std.math.maxInt(u32)) randomDevice() else params.seed);
    }

    // build the inference graph
    c.ggml_backend_sched_reset(ctx.sched);
    var built = try models.buildGraph(ctx, imgs, params);
    defer built.deinit();
    const gf = built.gf;
    _ = c.ggml_backend_sched_alloc_graph(ctx.sched, gf);

    // set inputs
    const model = &ctx.model;
    const hparams = &model.hparams;

    const image_size_width = imgs.entries.items[0].nx();
    const image_size_height = imgs.entries.items[0].ny();

    const patch_size = hparams.patch_size;
    const num_patches = @divTrunc(image_size_width, patch_size) * @divTrunc(image_size_height, patch_size);
    var st: EncodeState = .{
        .ctx = ctx,
        .params = params,
        .imgs = imgs,
        .gf = gf,
        .image_size_width = image_size_width,
        .image_size_height = image_size_height,
        .patch_size = patch_size,
        .num_patches = num_patches,
        .n_pos = num_patches + @as(c_int, if (model.class_embedding != null) 1 else 0),
        .pos_w = @divTrunc(image_size_width, patch_size),
        .pos_h = @divTrunc(image_size_height, patch_size),
    };

    // set input pixel values
    if (!imgs.is_audio) {
        var nelem: usize = 0;
        for (imgs.entries.items) |*img| nelem += @as(usize, @intCast(img.nx() * img.ny() * 3));
        const inp_raw = try gpa.alloc(f32, nelem);
        defer gpa.free(inp_raw);
        @memset(inp_raw, 0);

        // planar R, G, B per image; the batch dim is the temporal dim in Qwen-VL models
        {
            const nx: usize = @intCast(imgs.entries.items[0].nx());
            const ny: usize = @intCast(imgs.entries.items[0].ny());
            const n = nx * ny;

            var b: c_int = 0;
            while (b < n_batch_cur) : (b += 1) {
                impl.logDbg("%s: copying image %d/%d to input buffer (nx=%d, ny=%d)\n", .{ func, b + 1, n_batch_cur, @as(c_int, @intCast(nx)), @as(c_int, @intCast(ny)) });
                const buf = imgs.entries.items[@intCast(b)].roBuf();
                const batch_entry = inp_raw[@as(usize, @intCast(b)) * (3 * n) ..];
                for (0..ny) |y| {
                    for (0..nx) |x| {
                        const base_src = 3 * (y * nx + x);
                        const base_dst = y * nx + x;
                        batch_entry[base_dst] = buf[base_src];
                        batch_entry[1 * n + base_dst] = buf[base_src + 1];
                        batch_entry[2 * n + base_dst] = buf[base_src + 2];
                    }
                }
            }
        }
        st.setInputF32("inp_raw", inp_raw);
    } else if (params.gen_process != .gen_wav) {
        // audio input. GEN_WAV is not here: it takes codes or feats, set in the switch below
        impl.ggmlAssert(imgs.entries.items.len == 1, "clip.cpp", 4569, "imgs.entries.size() == 1");

        const mel_inp = &imgs.entries.items[0];
        const buf = mel_inp.roBuf();
        const n_step = mel_inp.nx();
        const n_mel = mel_inp.ny();
        impl.ggmlAssert(@as(usize, @intCast(n_step)) * @as(usize, @intCast(n_mel)) == buf.len, "clip.cpp", 4575, "(size_t)n_step * n_mel == buf.size()");

        st.setInputF32("inp_raw", buf);
    }

    // set input per projector; an arm that returns false is the C++'s
    // `return false` out of clip_encode
    if (!try arms_inputs.setInputsChecked(&st)) return false;

    // ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    const dev = c.ggml_backend_get_device(ctx.backend_cpu);
    const reg = if (dev != null) c.ggml_backend_dev_backend_reg(dev) else null;
    if (reg != null) {
        const fn_ptr = c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn_ptr != null) {
            const set_n_threads: c.ggml_backend_set_n_threads_t = @ptrCast(@alignCast(fn_ptr));
            set_n_threads.?(ctx.backend_cpu, params.n_threads);
        }
    }

    const status = c.ggml_backend_sched_graph_compute(ctx.sched, gf);
    if (status != c.GGML_STATUS_SUCCESS) {
        impl.logErr("%s: ggml_backend_sched_graph_compute failed with error %d\n", .{ func, @as(c_int, @intCast(status)) });
        return false;
    }

    // the last node is the embedding tensor, code2wav has no out_embd
    const embeddings: T = if (params.out_embd != null) c.ggml_graph_node(gf, -1) else null;

    if (embeddings != null) {
        // sanity check (assuming that all images in batch have the same number of tokens, so we only check the first one)
        const n_tokens_out: c_int = @intCast(embeddings.?.ne[1]);
        const expected_n_tokens_out = nOutputTokens(ctx, &imgs.entries.items[0]);
        if (n_tokens_out != expected_n_tokens_out) {
            impl.logErr("%s: expected output %d tokens, got %d\n", .{ func, expected_n_tokens_out, n_tokens_out });
            _ = c.ggml_abort("clip.cpp", 5764, "Invalid number of output tokens");
        }

        impl.logDbg("%s: output embedding shape [%d, %d, %d]\n", .{ func, @as(c_int, @intCast(embeddings.?.ne[0])), @as(c_int, @intCast(embeddings.?.ne[1])), @as(c_int, @intCast(embeddings.?.ne[2])) });

        // copy output to user buffer if provided
        // if output is empty, skip the copy
        const out_batch_embd = params.out_embd.?;
        if (out_batch_embd.items.len != 0) {
            if (out_batch_embd.items.len != @as(usize, @intCast(c.ggml_nelements(embeddings)))) {
                impl.logErr("%s: output buffer has %zu elements but expected %zu\n", .{ func, out_batch_embd.items.len, @as(usize, @intCast(c.ggml_nelements(embeddings))) });
                _ = c.ggml_abort("clip.cpp", 5776, "Output buffer size mismatch");
            }
            c.ggml_backend_tensor_get(embeddings, out_batch_embd.items.ptr, 0, c.ggml_nbytes(embeddings));
        } else {
            impl.logWrn("%s: output buffer is empty, skipping copy\n", .{func});
        }
    }

    //
    // for audio gen models
    //

    // optional outputs: a pipeline yields codes or feats, and not all have an eos head
    if (params.out_codes) |out_codes| {
        const codes = c.ggml_graph_get_tensor(gf, "out_codes");
        if (codes != null) {
            try out_codes.resize(gpa, @intCast(c.ggml_nelements(codes)));
            c.ggml_backend_tensor_get(codes, out_codes.items.ptr, 0, c.ggml_nbytes(codes));
        }
    }
    if (params.out_feats) |out_feats| {
        const feats = c.ggml_graph_get_tensor(gf, "out_feats");
        if (feats != null) {
            try out_feats.resize(gpa, @intCast(c.ggml_nelements(feats)));
            c.ggml_backend_tensor_get(feats, out_feats.items.ptr, 0, c.ggml_nbytes(feats));
        }
    }
    if (params.out_is_eos) |out_is_eos| {
        const eos = c.ggml_graph_get_tensor(gf, "out_eos_score");
        if (eos != null) {
            impl.ggmlAssert(c.ggml_nelements(eos) == 1, "clip.cpp", 5808, "ggml_nelements(eos) == 1");
            var score: f32 = 0.0;
            c.ggml_backend_tensor_get(eos, &score, 0, @sizeOf(f32));
            out_is_eos.* = score > hparams.gen_eos_threshold;
        }
    }
    if (params.out_audio) |out_audio| {
        const audio = c.ggml_graph_get_tensor(gf, "out_audio");
        if (audio == null) _ = c.ggml_abort("clip.cpp", 5817, "out_audio requested but graph has no \"out_audio\" tensor");
        try out_audio.resize(gpa, @intCast(c.ggml_nelements(audio)));
        c.ggml_backend_tensor_get(audio, out_audio.items.ptr, 0, c.ggml_nbytes(audio));

        // drop the tail audio that comes from the code-0 rear padding
        const n_codes: i64 = if (params.codes != null) graph_mod.ne(model.gen_code_head_w, 2) + 1 else 0;
        const n_frames_w: i64 = hparams.wav_tfm_swa;
        const n_frames: i64 = if (params.codes) |cd| @divTrunc(@as(i64, @intCast(cd.items.len)), n_codes) else n_frames_w;
        if (n_frames < n_frames_w) {
            const hop = out_audio.items.len / @as(usize, @intCast(n_frames_w));
            out_audio.shrinkRetainingCapacity(@as(usize, @intCast(n_frames)) * hop);
        }
    }
    if (params.state_out) |state_out| {
        const slots = try models.genStateSlots(hparams, model);
        defer models.freeSlots(slots);
        var total: usize = 0;
        for (slots) |slot| total += @as(usize, @intCast(slot.ne0 * slot.ne1)) * @sizeOf(f32);
        try state_out.resize(gpa, total);
        var offset: usize = 0;
        for (slots) |slot| {
            var nb: [256]u8 = undefined;
            _ = snprintf(&nb, nb.len, "state_out_%s", slot.name.ptr);
            const t = c.ggml_graph_get_tensor(gf, @ptrCast(&nb));
            if (t == null) _ = c.ggml_abort("clip.cpp", 5843, "state_out requested but graph has no \"state_out_%s\" tensor", slot.name.ptr);
            const n = c.ggml_nbytes(t);
            c.ggml_backend_tensor_get(t, state_out.items.ptr + offset, 0, n);
            offset += n;
        }
    }

    //
    // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set
    //

    if (ctx.debug_output_embeddings and embeddings != null) {
        const n_embd = embeddings.?.ne[0];
        const n_tokens = embeddings.?.ne[1];
        const emb_data = try gpa.alloc(f32, @intCast(c.ggml_nelements(embeddings)));
        defer gpa.free(emb_data);
        c.ggml_backend_tensor_get(embeddings, emb_data.ptr, 0, c.ggml_nbytes(embeddings));

        impl.logInf("\n=== MTMD_DEBUG_EMBEDDINGS ===\n", .{});
        impl.logInf("Shape: [%lld, %lld]\n", .{ @as(c_longlong, n_embd), @as(c_longlong, n_tokens) });

        // TEMP debugging (parity validation), will be removed before merge
        // when the env var holds a path, dump the raw data: [int32 n_tokens][int32 n_embd][f32 data]
        if (std.c.getenv("MTMD_DEBUG_EMBEDDINGS")) |dump_path| {
            if (!std.mem.eql(u8, std.mem.span(dump_path), "1")) {
                if (std.c.fopen(dump_path, "wb")) |f| {
                    const hdr = [2]i32{ @truncate(n_tokens), @truncate(n_embd) };
                    _ = std.c.fwrite(std.mem.asBytes(&hdr), @sizeOf(@TypeOf(hdr)), 1, f);
                    _ = std.c.fwrite(@ptrCast(emb_data.ptr), @sizeOf(f32), emb_data.len, f);
                    _ = std.c.fclose(f);
                }
            }
        }

        // Print first few values of first token
        impl.logInf("Token 0 (first 16 values): ", .{});
        var i: i64 = 0;
        while (i < @min(@as(i64, 16), n_embd)) : (i += 1) impl.logInf("%.6f ", .{@as(f64, emb_data[@intCast(i)])});
        impl.logInf("\n", .{});

        // Print last few values of first token
        if (n_embd > 16) {
            impl.logInf("Token 0 (last 16 values):  ", .{});
            i = n_embd - 16;
            while (i < n_embd) : (i += 1) impl.logInf("%.6f ", .{@as(f64, emb_data[@intCast(i)])});
            impl.logInf("\n", .{});
        }

        // Compute and print statistics
        var sum: f32 = 0.0;
        var sum_sq: f32 = 0.0;
        var min_val = emb_data[0];
        var max_val = emb_data[0];
        for (emb_data) |v| {
            sum += v;
            sum_sq = @mulAdd(f32, v, v, sum_sq);
            min_val = @min(min_val, v);
            max_val = @max(max_val, v);
        }
        const nf: f32 = @floatFromInt(emb_data.len);
        const mean = sum / nf;
        const variance = @mulAdd(f32, -mean, mean, sum_sq / nf);
        impl.logInf("Stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, sum=%.6f\n", .{ @as(f64, mean), @as(f64, sqrtf(variance)), @as(f64, min_val), @as(f64, max_val), @as(f64, sum) });
        impl.logInf("=== END MTMD_DEBUG_EMBEDDINGS ===\n\n", .{});
    }

    return true;
}

extern fn sqrtf(x: f32) f32;

/// Ports `clip_n_mmproj_embd` (mtmd/clip.cpp:5911 @b29c606e2): the embedding
/// size of the projector's output, which must be the text model's; the
/// per-projector body is `arms_tensors.nMmprojEmbd`.
pub fn nMmprojEmbd(ctx: *const Ctx) c_int {
    return arms_tensors.nMmprojEmbd(ctx);
}

/// Ports `clip_is_llava` (mtmd/clip.cpp:6015 @b29c606e2).
pub fn isLlava(ctx: *const Ctx) bool {
    return ctx.model.hparams.has_llava_projector;
}

/// Ports `clip_has_vision_encoder` (mtmd/clip.cpp:6019 @b29c606e2).
pub fn hasVisionEncoder(ctx: *const Ctx) bool {
    return ctx.model.modality == .vision;
}

/// Ports `clip_has_audio_encoder` (mtmd/clip.cpp:6023 @b29c606e2).
pub fn hasAudioEncoder(ctx: *const Ctx) bool {
    return ctx.model.modality == .audio;
}

/// Ports `clip_support_batch` (mtmd/clip.cpp:6027 @b29c606e2).
pub fn supportBatch(ctx: *const Ctx) bool {
    return ctx.support_batch;
}

/// Ports `clip_model_n_temporal_merge` (mtmd/clip.cpp:6034 @b29c606e2): the
/// qwen-vl models fuse two frames into one.
pub fn modelNTemporalMerge(ctx: *const Ctx) c_int {
    return switch (ctx.projType()) {
        .qwen2vl, .qwen25vl, .qwen3vl => 2,
        else => 1,
    };
}

/// Ports `clip_get_projector_type` (mtmd/clip.cpp:6049 @b29c606e2).
pub fn getProjectorType(ctx: *const Ctx) names.ProjectorType {
    return ctx.projType();
}

/// Ports `clip_get_hparams` (mtmd/clip.cpp:6053 @b29c606e2).
pub fn getHparams(ctx: *const Ctx) *const model_mod.Hparams {
    return &ctx.model.hparams;
}

/// Ports `clip_get_mem_usage` (mtmd/clip.cpp:6057 @b29c606e2): weights plus
/// compute, by device. The caller owns the map.
pub fn getMemUsage(ctx: *const Ctx) Error!DevSizeMap {
    var res: DevSizeMap = .{};
    for (ctx.mem_usage.items.items) |e| try res.add(e.dev, e.size);
    for (ctx.mem_compute.items.items) |e| try res.add(e.dev, e.size);
    return res;
}

/// Ports `clip_set_debug_output_embeddings` (mtmd/clip.cpp:6069 @b29c606e2).
pub fn setDebugOutputEmbeddings(ctx: *Ctx, enable: bool) void {
    ctx.debug_output_embeddings = enable;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    impl.refAllDeep(@This());
}
