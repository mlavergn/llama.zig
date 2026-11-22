//! The mtmd model and subcontext: loading a projector from GGUF into an
//! `mtmd_model`, and building and running its graphs in an
//! `mtmd_subcontext` on the backends every subcontext shares.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/mtmd/mtmd-model.cpp` (`mtmd_model_loader`,
//! `mtmd_model_load`, `mtmd_model_get_caps`, `mtmd_model`'s methods) and
//! `llama.cpp/tools/mtmd/mtmd-subcontext.cpp` (`mtmd_backends`,
//! `mtmd_subcontext`, the encoder and generator states), with the types of
//! `llama.cpp/tools/mtmd/mtmd-subcontext.h`, at v1.0.0 (`b1b770d90`). The
//! backends and the scheduler are one object shared by the subcontexts
//! (a591cbe17). The per-model graphs under `models/` import this file, so it
//! keeps the name `clip.zig`. The four per-projector switches live beside it:
//!
//! - `arms_hparams.zig` -- `load_hparams`'s model-specific switch
//! - `arms_tensors.zig` -- `load_tensors`'s model-specific switch and
//!   `mtmd_model::n_mmproj_embd`
//! - `arms_inputs.zig` -- `mtmd_subcontext::encode`'s per-projector input
//!   switch and `mtmd_model::n_output_tokens`
//! - `models/module.zig` -- `mtmd_get_graph_builder` and the graphs
//!
//! Each declaration names what it ports and the line it began at.
//!
//! # Exceptions
//!
//! Upstream's loader throws `std::runtime_error`, and `mtmd_init_from_file`
//! catches it and logs `what()`. Here a throw is `throwErr`, which records the
//! message in `what_buf` and returns `error.Runtime`.

const std = @import("std");
const mod = @import("module.zig");
const stdlib = @import("stdlib");

pub const Error = error{ Runtime, OutOfMemory };

/// `e.what()` of the last throw.
pub var what_buf: [2048]u8 = undefined;
pub var what_len: usize = 0;

/// `throw std::runtime_error(mtmd_format(fmt, args...))`.
pub fn throwErr(comptime fmt: [*:0]const u8, args: anytype) Error {
    const n = @call(.auto, stdlib.libc.snprintf, .{ &what_buf, what_buf.len, fmt } ++ args);
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

/// Ports `mtmd_model_caps` (mtmd-model.h:1326 @b1b770d90).
pub const ModelCaps = struct { has_vision: bool, has_audio: bool, has_gen_audio: bool };

/// Ports `mtmd_subctx_enc_state` (mtmd-subcontext.h:76 @b1b770d90): the
/// persistent state of a causal audio encoder, for streaming input (one per
/// stream); see `Ctx.encStateInit`.
pub const EncState = struct {
    n_ctx: i32 = 0, // ring slots: attention window + frames per call
    n_past: i32 = 0, // absolute position of the next frame

    slot_pos: std.ArrayList(i32) = .empty, // absolute position held by each slot, -1 if empty
    stem_tail: std.ArrayList(f32) = .empty, // last conv stem output frame, empty on a cold start

    ctx_data: ?*mod.impl.c.ggml_context = null,
    buf: mod.impl.c.ggml_backend_buffer_t = null,
    k: std.ArrayList(*mod.impl.c.ggml_tensor) = .empty, // per layer [n_embd_kv, n_ctx]
    v: std.ArrayList(*mod.impl.c.ggml_tensor) = .empty,

    // lfm3a: same order as list_lfm3a_state_slots(), F32 [ne0, ne1], updated in place
    slots: std.ArrayList(*mod.impl.c.ggml_tensor) = .empty,

    // nemotron transcribe: the window's normalized log-mel, cache and FIFO rows in; its speaker rows and predictions out
    mel_norm: std.ArrayList(f32) = .empty,
    spk_prev: std.ArrayList(f32) = .empty,
    spk_chunk: std.ArrayList(f32) = .empty,
    spk_preds: std.ArrayList(f32) = .empty,

    /// The C++'s destructor (`delete`), members in reverse order.
    pub fn destroy(self: *EncState) void {
        self.spk_preds.deinit(std.heap.c_allocator);
        self.spk_chunk.deinit(std.heap.c_allocator);
        self.spk_prev.deinit(std.heap.c_allocator);
        self.mel_norm.deinit(std.heap.c_allocator);
        self.slots.deinit(std.heap.c_allocator);
        self.v.deinit(std.heap.c_allocator);
        self.k.deinit(std.heap.c_allocator);
        if (self.buf != null) mod.impl.c.ggml_backend_buffer_free(self.buf);
        if (self.ctx_data != null) mod.impl.c.ggml_free(self.ctx_data);
        self.stem_tail.deinit(std.heap.c_allocator);
        self.slot_pos.deinit(std.heap.c_allocator);
        std.heap.c_allocator.destroy(self);
    }
};

/// Ports `mtmd_subctx_gen_state` (mtmd-subcontext.h:98 @b1b770d90): the
/// per-sequence state of audio generation, the RNG and the slots kept
/// between GEN_WAV calls; see `Ctx.genStateInit`.
pub const GenState = struct {
    rng: stdlib.libcxx.Mt19937 = undefined,

    ctx_data: ?*mod.impl.c.ggml_context = null,
    buf: mod.impl.c.ggml_backend_buffer_t = null,
    slots: std.ArrayList(*mod.impl.c.ggml_tensor) = .empty, // same order as list_gen_state_slots(), F32 [ne0, ne1]

    /// Ports `reset` (mtmd-subcontext.h:106 @b1b770d90), a member of
    /// `mtmd_subctx_gen_state`: clears the slots and keeps the RNG stream.
    pub fn reset(self: *GenState) void {
        if (self.buf != null) mod.impl.c.ggml_backend_buffer_clear(self.buf, 0);
    }

    /// The C++'s destructor (`delete`), members in reverse order.
    pub fn destroy(self: *GenState) void {
        self.slots.deinit(std.heap.c_allocator);
        if (self.buf != null) mod.impl.c.ggml_backend_buffer_free(self.buf);
        if (self.ctx_data != null) mod.impl.c.ggml_free(self.ctx_data);
        std.heap.c_allocator.destroy(self);
    }
};

/// Ports `mtmd_subctx_encode_params` (mtmd-subcontext.h:113 @b1b770d90).
pub const EncodeParams = struct {
    n_threads: c_int = 1,
    imgs: ?*const mod.impl.ImageF32Batch = null,
    out_embd: ?*std.ArrayList(f32) = null,

    // for audio gen, imgs has exactly one entry: hidden state from backbone (GEN_CODE) or unused (GEN_WAV)
    gen_process: mod.impl.GenProcessType = .unknown,

    // GEN_CODE: out_embd receives the embd to feed back to the backbone
    code0: i32 = 0, // semantic code sampled by the backbone
    top_k: i32 = 50,
    top_p: f32 = 1.0,
    out_codes: ?*std.ArrayList(i32) = null, // this frame's 16 sampled codes
    out_feats: ?*std.ArrayList(f32) = null, // continuous counterpart of out_codes
    temp: f32 = 0.0, // sampling temperature, noise scale for flow-matching decoders
    out_is_eos: ?*bool = null,

    // TOKENIZE: speech embedding rows of the produced codes
    out_code_embd: ?*std.ArrayList(f32) = null,

    // GEN_WAV
    codes: ?*const std.ArrayList(i32) = null, // this frame's 16 RVQ codes
    feats: ?*const std.ArrayList(f32) = null, // continuous counterpart of codes
    out_audio: ?*std.ArrayList(f32) = null, // decoded PCM samples, F32
    ref: ?*const mod.impl.GenRef = null, // TTS reference conditioning of the voice
    // TTS_VOCODE internal stage inputs
    mel_in: ?*const std.ArrayList(f32) = null,
    sstft_in: ?*const std.ArrayList(f32) = null,
    vocode_n_mel: c_int = 0,
    vocode_n_stft: c_int = 0,
    out_spec: ?*std.ArrayList(f32) = null,

    // audio gen: state of one sequence, encode() updates it in place
    gen_state: ?*GenState = null,

    // streaming audio input, updated in place by encode()
    enc_state: ?*EncState = null,
};

/// Ports `mtmd_subctx_params` (mtmd-subcontext.h:27 @b1b770d90).
pub const SubctxParams = struct {
    flash_attn_type: mod.impl.FlashAttnType,
    warmup: bool,
};

// -----------------------------------------------------------------------------
// mtmd_backends

/// Ports `mtmd_backend_cpu_set_n_threads` (mtmd-subcontext.cpp:24
/// @b1b770d90): `ggml_backend_cpu_set_n_threads(backend_cpu, n_threads)`
/// through the registry.
pub fn backendCpuSetNThreads(backend_cpu: mod.impl.c.ggml_backend_t, n_threads: c_int) void {
    const dev = mod.impl.c.ggml_backend_get_device(backend_cpu);
    const reg = if (dev != null) mod.impl.c.ggml_backend_dev_backend_reg(dev) else null;
    if (reg != null) {
        const fn_ptr = mod.impl.c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn_ptr != null) {
            const set_n_threads: mod.impl.c.ggml_backend_set_n_threads_t = @ptrCast(@alignCast(fn_ptr));
            set_n_threads.?(backend_cpu, n_threads);
        }
    }
}

/// Ports `mtmd_threadpool_init` (mtmd-subcontext.cpp:35 @b1b770d90): a CPU
/// threadpool for two threads or more, through the registry; null if the
/// CPU backend has none to give. The `shared_ptr`'s deleter is `free_fn`,
/// kept beside the pool.
fn threadpoolInit(n_threads: c_int, free_out: *?*const fn (mod.impl.c.ggml_threadpool_t) callconv(.c) void) Error!mod.impl.c.ggml_threadpool_t {
    const dev = mod.impl.c.ggml_backend_dev_by_type(mod.impl.c.GGML_BACKEND_DEVICE_TYPE_CPU);
    const reg = if (dev != null) mod.impl.c.ggml_backend_dev_backend_reg(dev) else null;
    if (reg == null or n_threads < 2) return null;
    const new_ptr = mod.impl.c.ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    const free_ptr = mod.impl.c.ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    if (new_ptr == null or free_ptr == null) return null;
    const new_fn: *const fn ([*c]mod.impl.c.ggml_threadpool_params) callconv(.c) mod.impl.c.ggml_threadpool_t = @ptrCast(@alignCast(new_ptr));
    const free_fn: *const fn (mod.impl.c.ggml_threadpool_t) callconv(.c) void = @ptrCast(@alignCast(free_ptr));
    var tpp = mod.impl.c.ggml_threadpool_params_default(n_threads);
    tpp.poll = 0; // the pool is idle most of the time, don't take CPU away from the text model
    const threadpool = new_fn(&tpp);
    if (threadpool == null) return throwErr("failed to create threadpool", .{});
    free_out.* = free_fn;
    return threadpool;
}

/// Ports `mtmd_select_device` (mtmd-subcontext.cpp:55 @b1b770d90): the
/// device the weights and the graphs are placed on, null for the CPU.
pub fn selectDevice(use_gpu: bool, device: mod.impl.c.ggml_backend_dev_t) mod.impl.c.ggml_backend_dev_t {
    if (!use_gpu) return null;
    if (device != null) return device;
    const dev = mod.impl.c.ggml_backend_dev_by_type(mod.impl.c.GGML_BACKEND_DEVICE_TYPE_GPU);
    return if (dev != null) dev else mod.impl.c.ggml_backend_dev_by_type(mod.impl.c.GGML_BACKEND_DEVICE_TYPE_IGPU);
}

/// Ports `mtmd_backends` (mtmd-subcontext.h:36 @b1b770d90): the backends and
/// the scheduler every subcontext of one `mtmd_context` shares (a591cbe17), so its compute buffers fit the largest graph any of them
/// reserved. Not thread-safe.
pub const Backends = struct {
    backend: mod.impl.c.ggml_backend_t = null, // backend_cpu if no accelerator
    backend_cpu: mod.impl.c.ggml_backend_t = null,

    // for ggml_backend_sched_new(), backend_cpu last
    ptrs: std.ArrayList(mod.impl.c.ggml_backend_t) = .empty,
    bufts: std.ArrayList(mod.impl.c.ggml_backend_buffer_type_t) = .empty,

    threadpool: mod.impl.c.ggml_threadpool_t = null, // can be null
    threadpool_free: ?*const fn (mod.impl.c.ggml_threadpool_t) callconv(.c) void = null,

    // its buffers fit the largest graph reserved by the subcontexts
    sched: mod.impl.c.ggml_backend_sched_t = null,

    /// Ports `mtmd_backends` (mtmd-subcontext.cpp:66 @b1b770d90), its
    /// constructor; `device` null means the CPU.
    pub fn create(device: mod.impl.c.ggml_backend_dev_t, n_threads: c_int, max_nodes: c_int, cb_eval: mod.impl.c.ggml_backend_sched_eval_callback, cb_eval_user_data: ?*anyopaque) Error!*Backends {
        const func = "mtmd_backends";
        const self = try std.heap.c_allocator.create(Backends);
        self.* = .{};
        // a throwing constructor runs the destructors of the members built so far
        errdefer self.destroyMembers();
        self.threadpool = try threadpoolInit(n_threads, &self.threadpool_free);

        self.backend_cpu = mod.impl.c.ggml_backend_init_by_type(mod.impl.c.GGML_BACKEND_DEVICE_TYPE_CPU, null);
        if (self.backend_cpu == null) return throwErr("failed to initialize CPU backend", .{});
        if (self.threadpool != null) {
            const reg = mod.impl.c.ggml_backend_dev_backend_reg(mod.impl.c.ggml_backend_get_device(self.backend_cpu));
            const p = mod.impl.c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
            if (p != null) {
                const set_threadpool_fn: *const fn (mod.impl.c.ggml_backend_t, mod.impl.c.ggml_threadpool_t) callconv(.c) void = @ptrCast(@alignCast(p));
                set_threadpool_fn(self.backend_cpu, self.threadpool);
            }
        }
        if (device != null) {
            self.backend = mod.impl.c.ggml_backend_dev_init(device, null);
            if (self.backend == null) {
                mod.impl.c.ggml_backend_free(self.backend_cpu);
                self.backend_cpu = null;
                return throwErr("%s: failed to initialize \"%s\" backend\n", .{ func, mod.impl.c.ggml_backend_dev_name(device) });
            }
        }

        if (self.backend != null) {
            mod.impl.logInf("%s: using %s backend\n", .{ func, mod.impl.c.ggml_backend_name(self.backend) });
            try self.ptrs.append(std.heap.c_allocator, self.backend);
            try self.bufts.append(std.heap.c_allocator, mod.impl.c.ggml_backend_get_default_buffer_type(self.backend));
        } else {
            self.backend = self.backend_cpu;
            mod.impl.logInf("%s: using CPU backend\n", .{func});
        }

        try self.ptrs.append(std.heap.c_allocator, self.backend_cpu);
        try self.bufts.append(std.heap.c_allocator, mod.impl.c.ggml_backend_get_default_buffer_type(self.backend_cpu));

        self.sched = mod.impl.c.ggml_backend_sched_new(self.ptrs.items.ptr, self.bufts.items.ptr, @intCast(self.ptrs.items.len), @intCast(max_nodes), false, true);
        if (cb_eval != null) mod.impl.c.ggml_backend_sched_set_eval_callback(self.sched, cb_eval, cb_eval_user_data);
        return self;
    }

    fn destroyMembers(self: *Backends) void {
        if (self.sched != null) mod.impl.c.ggml_backend_sched_free(self.sched);
        if (self.threadpool != null) self.threadpool_free.?(self.threadpool);
        self.bufts.deinit(std.heap.c_allocator);
        self.ptrs.deinit(std.heap.c_allocator);
        std.heap.c_allocator.destroy(self);
    }

    /// Ports `mtmd_backends` (mtmd-subcontext.cpp:108 @b1b770d90), its destructor: the
    /// scheduler before the backends it uses, then the members in reverse
    /// order (the threadpool after the CPU backend that ran on it).
    pub fn destroy(self: *Backends) void {
        // before the backends it uses
        if (self.sched != null) mod.impl.c.ggml_backend_sched_free(self.sched);
        self.sched = null;
        mod.impl.c.ggml_backend_free(self.backend);
        if (self.backend != self.backend_cpu) mod.impl.c.ggml_backend_free(self.backend_cpu);
        self.destroyMembers();
    }

    /// Ports `get_mem_compute` (mtmd-subcontext.cpp:117 @b1b770d90), a member
    /// of `mtmd_backends`. The caller owns the map.
    pub fn getMemCompute(self: *const Backends) Error!mod.model.DevSizeMap {
        var res: mod.model.DevSizeMap = .{};
        errdefer res.deinit();
        for (self.ptrs.items) |b| try res.add(mod.impl.c.ggml_backend_get_device(b), mod.impl.c.ggml_backend_sched_get_buffer_size(self.sched, b));
        return res;
    }
};

// -----------------------------------------------------------------------------
// mtmd_subcontext

/// Ports `mtmd_subcontext` (mtmd-subcontext.h:152 @b1b770d90): runs the graphs
/// of one `mtmd_model` on the shared backends and scheduler.
pub const Ctx = struct {
    model: *const mod.model.Model,
    backends: *Backends,

    buf_compute_meta: std.ArrayList(u8) = .empty,

    max_nodes: c_int = 8192,
    flash_attn_type: mod.impl.FlashAttnType = .auto,
    is_allocated: bool = false,

    debug_output_embeddings: bool = false,

    support_batch: bool = false,

    /// Ports `mtmd_subcontext` (mtmd-subcontext.cpp:125 @b1b770d90), its
    /// constructor. `model` and `backends` must outlive the subcontext.
    pub fn create(model: *const mod.model.Model, backends: *Backends, params: SubctxParams) Error!*Ctx {
        const self = try std.heap.c_allocator.create(Ctx);
        self.* = .{ .model = model, .backends = backends, .max_nodes = model.max_nodes, .flash_attn_type = params.flash_attn_type };
        errdefer self.destroy();
        if (self.flash_attn_type == .auto and !model.flash_attn_auto) {
            self.flash_attn_type = .disabled;
        }

        self.debug_output_embeddings = stdlib.env.getenv("MTMD_DEBUG_EMBEDDINGS") != null;

        try self.buf_compute_meta.resize(std.heap.c_allocator, @as(usize, @intCast(self.max_nodes)) * mod.impl.c.ggml_tensor_overhead() + mod.impl.c.ggml_graph_overhead_custom(@intCast(self.max_nodes), false));

        var batch = try self.getDummyBatch();
        defer batch.deinit();
        if (model.modality != .gen_audio) {
            // check batching support: the C++ constructs a builder for the
            // dummy batch and asks it, without building
            self.support_batch = mod.models.supportBatch(self.projType()) orelse mod.impl.abort("mtmd-graph.cpp", 1056, "missing cgraph builder");
        }
        if (params.warmup) {
            try self.warmup(&batch);
        }
        return self;
    }

    /// The C++'s destructor.
    pub fn destroy(self: *Ctx) void {
        self.buf_compute_meta.deinit(std.heap.c_allocator);
        std.heap.c_allocator.destroy(self);
    }

    pub fn projType(self: *const Ctx) mod.names.ProjectorType {
        return self.model.proj_type;
    }

    /// What a `graph.Graph` is built from.
    pub fn graphSource(self: *Ctx) mod.graph.Source {
        return .{
            .model = self.model,
            .buf_compute_meta = self.buf_compute_meta.items,
            .max_nodes = self.max_nodes,
            .flash_attn_type = self.flash_attn_type,
            .n_mmproj_embd = nMmprojEmbd(self.model),
        };
    }

    /// Ports `get_dummy_batch` (mtmd-subcontext.cpp:148 @b1b770d90), a member
    /// of `mtmd_subcontext`: a fake batch.
    pub fn getDummyBatch(self: *const Ctx) Error!mod.impl.ImageF32Batch {
        const func = "get_dummy_batch";
        const model = self.model;
        const hparams = &model.hparams;
        var batch: mod.impl.ImageF32Batch = .{};
        errdefer batch.deinit();
        var img: mod.impl.ImageF32 = .{};
        errdefer img.deinit();
        if (model.modality == .vision) {
            const sz = hparams.warmup_image_size;
            try img.setSize(.{ .width = sz, .height = sz }, false, false);
            mod.impl.logInf("%s: warmup with image size = %d x %d\n", .{ func, sz, sz });
        } else if (model.modality == .gen_audio) {
            // TODO: improve this, move to per-model impl
            const n_embd = nMmprojEmbd(model);
            try img.setSize(.{ .width = n_embd, .height = 1 }, false, true);
            mod.impl.logInf("%s: warmup with one gen-audio step\n", .{func});
        } else {
            // GEMMA4UA uses n_mel_bins as a raw-waveform frame size (640), not a mel-bin count,
            // so the [1, 256] bound only applies to FFT-based models.
            const fft_based = model.proj_type != .gemma4ua;
            if (hparams.n_mel_bins <= 0 or (fft_based and hparams.n_mel_bins > 256)) {
                return throwErr("%s: invalid n_mel_bins (%d), must be in [1, 256]\n", .{ func, hparams.n_mel_bins });
            }
            try img.setSize(.{ .width = hparams.warmup_audio_size, .height = hparams.n_mel_bins }, false, false);
            mod.impl.logInf("%s: warmup with audio size = %d\n", .{ func, hparams.warmup_audio_size });
        }
        try batch.entries.append(std.heap.c_allocator, img);
        return batch;
    }

    const SupportInfoOp = struct {
        op: mod.graph.T,
        // true if the op runs on the accelerated backend
        is_accel: bool = true,
    };

    const SupportInfoGraph = struct {
        // whether the backend supports flash attention
        fattn: bool = true,
        fattn_op: mod.graph.T = null, // for debugging
        ops: std.ArrayList(SupportInfoOp) = .empty,

        fn deinit(self: *SupportInfoGraph) void {
            self.ops.deinit(std.heap.c_allocator);
        }
    };

    /// Ports `warmup` (mtmd-subcontext.cpp:176 @b1b770d90), a member of
    /// `mtmd_subcontext`: reserve the compute buffers for `batch`.
    pub fn warmup(self: *Ctx, batch: *const mod.impl.ImageF32Batch) Error!void {
        const func = "warmup";
        const backends = self.backends;
        var info: SupportInfoGraph = .{};
        defer info.deinit();

        if (self.flash_attn_type == .auto) {
            // try to enable flash attention to see if it's supported
            self.flash_attn_type = .enabled;
            info.deinit();
            info = try self.reserveComputeMeta(batch);
            if (!info.fattn and info.fattn_op != null) {
                const op = info.fattn_op;
                mod.impl.logWrn("%s: *****************************************************************\n", .{func});
                mod.impl.logWrn("%s: WARNING: flash attention not supported by %s, memory usage will increase\n", .{ func, mod.impl.c.ggml_backend_name(backends.backend) });
                mod.impl.logWrn("%s: op params: \n", .{func});
                printShape(func, " dst", op);
                printShape(func, "src0", op.?.src[0]);
                printShape(func, "src1", op.?.src[1]);
                printShape(func, "src2", op.?.src[2]);
                mod.impl.logWrn("%s: please report this on github as an issue\n", .{func});
                mod.impl.logWrn("%s: *****************************************************************\n", .{func});
                self.flash_attn_type = .disabled;
                var info2 = try self.reserveComputeMeta(batch);
                info2.deinit();
            }
        } else {
            info.deinit();
            info = try self.reserveComputeMeta(batch);
            if (!info.fattn and self.flash_attn_type == .enabled) {
                mod.impl.logWrn("%s: flash attention is not supported by the current backend; falling back to CPU (performance will be degraded)\n", .{func});
            }
        }

        self.is_allocated = true; // mark buffers as allocated

        mod.impl.logInf("%s: flash attention is %s\n", .{ func, @as([*:0]const u8, if (self.flash_attn_type == .enabled) "enabled" else "disabled") });

        // print ops that are not supported by the GPU backend (if there is one)
        if (backends.backend != backends.backend_cpu) {
            var unsupported_ops: std.ArrayList(SupportInfoOp) = .empty;
            defer unsupported_ops.deinit(std.heap.c_allocator);
            for (info.ops.items) |op| {
                if (!op.is_accel) try unsupported_ops.append(std.heap.c_allocator, op);
            }
            if (unsupported_ops.items.len != 0) {
                mod.impl.logWrn("%s: *****************************************************************\n", .{func});
                mod.impl.logWrn("%s: WARNING: the mmproj graph uses unsupported operators by the backend\n", .{func});
                mod.impl.logWrn("%s:          the performance will be suboptimal                      \n", .{func});
                mod.impl.logWrn("%s:          list of unsupported ops (backend=%s):\n", .{ func, mod.impl.c.ggml_backend_name(backends.backend) });
                for (unsupported_ops.items) |op| {
                    const t = op.op.?;
                    mod.impl.logWrn("%s: %16s: type = %s, ne = [%d %d %d %d]\n", .{
                        func,                           mod.impl.c.ggml_op_name(t.op),  mod.impl.c.ggml_type_name(t.type),
                        @as(c_int, @truncate(t.ne[0])), @as(c_int, @truncate(t.ne[1])), @as(c_int, @truncate(t.ne[2])),
                        @as(c_int, @truncate(t.ne[3])),
                    });
                }
                mod.impl.logWrn("%s: flash attention is %s\n", .{ func, @as([*:0]const u8, if (self.flash_attn_type == .enabled) "enabled" else "disabled") });
                mod.impl.logWrn("%s: please report this on github as an issue\n", .{func});
                mod.impl.logWrn("%s: ref: https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118\n", .{func});
                mod.impl.logWrn("%s: *****************************************************************\n", .{func});
            }
        }
    }

    /// The `print_shape` lambda of `warmup` (mtmd-subcontext.cpp:176
    /// @b1b770d90): `%d` over `int64_t` and `size_t`, which the C++ passes as
    /// 64-bit values and `printf` reads as their low 32 bits on this ABI.
    fn printShape(func: [*:0]const u8, name: [*:0]const u8, t: mod.graph.T) void {
        const x = t.?;
        mod.impl.logWrn("%s:   %s: type = %s, ne = [%d %d %d %d], nb = [%d %d %d %d]\n", .{
            func,                                               name,                                               mod.impl.c.ggml_type_name(x.type),
            @as(c_int, @truncate(x.ne[0])),                     @as(c_int, @truncate(x.ne[1])),                     @as(c_int, @truncate(x.ne[2])),
            @as(c_int, @truncate(x.ne[3])),                     @as(c_int, @truncate(@as(i64, @bitCast(x.nb[0])))), @as(c_int, @truncate(@as(i64, @bitCast(x.nb[1])))),
            @as(c_int, @truncate(@as(i64, @bitCast(x.nb[2])))), @as(c_int, @truncate(@as(i64, @bitCast(x.nb[3])))),
        });
    }

    /// Ports `reserve_compute_meta` (mtmd-subcontext.cpp:243 @b1b770d90), a
    /// member of `mtmd_subcontext`: initialize the backend buffers, without
    /// allocating them. The scheduler is shared, so the compute
    /// buffer sizes are logged once, by `mtmd_context`, not per reservation.
    fn reserveComputeMeta(self: *Ctx, batch: *const mod.impl.ImageF32Batch) Error!SupportInfoGraph {
        const func = "reserve_compute_meta";
        const sched = self.backends.sched;
        var built = try mod.models.buildGraph(self, batch, null);
        defer built.deinit();
        const gf = built.gf;
        _ = mod.impl.c.ggml_backend_sched_reserve(sched, gf);

        const n_splits = mod.impl.c.ggml_backend_sched_get_n_splits(sched);
        const n_nodes = mod.impl.c.ggml_graph_n_nodes(gf);

        mod.impl.logInf("%s: graph splits = %d, nodes = %d\n", .{ func, n_splits, n_nodes });

        var res: SupportInfoGraph = .{};
        errdefer res.deinit();

        // check op support
        var i: c_int = 0;
        while (i < mod.impl.c.ggml_graph_n_nodes(gf)) : (i += 1) {
            const node = mod.impl.c.ggml_graph_node(gf, i);
            try res.ops.append(std.heap.c_allocator, .{ .op = node, .is_accel = true });
            if (!mod.impl.c.ggml_backend_supports_op(self.backends.backend, node)) {
                res.ops.items[res.ops.items.len - 1].is_accel = false;
                if (node.*.op == mod.impl.c.GGML_OP_FLASH_ATTN_EXT) {
                    res.fattn = false;
                    res.fattn_op = node;
                }
            }
        }
        return res;
    }

    /// Ports `encode_image` (mtmd-subcontext.cpp:368 @b1b770d90), a member of
    /// `mtmd_subcontext`.
    pub fn encodeImage(self: *Ctx, n_threads: c_int, img: *const mod.impl.ImageF32, out_vec: *std.ArrayList(f32)) Error!bool {
        var imgs: mod.impl.ImageF32Batch = .{};
        defer imgs.deinit();
        try imgs.entries.append(std.heap.c_allocator, try img.clone());
        return self.encodeBatch(n_threads, &imgs, out_vec);
    }

    /// Ports `encode_batch` (mtmd-subcontext.cpp:376 @b1b770d90), a member of
    /// `mtmd_subcontext`.
    pub fn encodeBatch(self: *Ctx, n_threads: c_int, imgs: *const mod.impl.ImageF32Batch, out_batch_embd: *std.ArrayList(f32)) Error!bool {
        var params: EncodeParams = .{ .imgs = imgs, .n_threads = n_threads, .out_embd = out_batch_embd };
        if (self.projType() == .nemotron_transcribe) {
            return mod.models.nemotron_transcribe.encodeWindows(self, self.model, &params);
        }
        return encode(self, &params);
    }
};

// -----------------------------------------------------------------------------
// mtmd_model_loader

/// Ports `mtmd_model_loader` (mtmd-model.cpp:32 @b1b770d90).
pub const Loader = struct {
    ctx_meta: ?*mod.impl.c.ggml_context = null,
    ctx_gguf: ?*mod.impl.c.gguf_context = null,

    fname: [:0]const u8,

    model_size: usize = 0, // in bytes

    has_vision: bool = false,
    has_audio: bool = false,
    has_gen_audio: bool = false,

    progress_callback: ?*const fn (f32, ?*anyopaque) callconv(.c) bool = null,
    progress_callback_user_data: ?*anyopaque = null,

    /// Ports `mtmd_model_loader` (mtmd-model.cpp:47 @b1b770d90), the constructor of `mtmd_model_loader`.
    pub fn init(fname: [:0]const u8, skip_tensors: bool, progress_cb: ?*const fn (f32, ?*anyopaque) callconv(.c) bool, progress_user_data: ?*anyopaque) Error!Loader {
        const func = "mtmd_model_loader";
        var self: Loader = .{ .fname = fname, .progress_callback = progress_cb, .progress_callback_user_data = progress_user_data };
        var meta: ?*mod.impl.c.ggml_context = null;
        const params: mod.impl.c.gguf_init_params = .{ .no_alloc = true, .ctx = &meta };

        self.ctx_gguf = mod.impl.c.gguf_init_from_file(fname.ptr, params);
        if (self.ctx_gguf == null) return throwErr("%s: failed to load mmproj model from %s. Does this file exist?\n", .{ func, fname.ptr });

        self.ctx_meta = meta;

        const n_tensors: c_int = @intCast(mod.impl.c.gguf_get_n_tensors(self.ctx_gguf));

        // print gguf info
        {
            var name_buf: ?[*:0]const u8 = "";
            var desc_buf: ?[*:0]const u8 = "";
            try self.getStringPtr(mod.names.KEY_NAME, &name_buf, false);
            try self.getStringPtr(mod.names.KEY_DESCRIPTION, &desc_buf, false);
            mod.impl.logInf("%s: model name:   %s\n", .{ func, name_buf.? });
            mod.impl.logInf("%s: description:  %s\n", .{ func, desc_buf.? });
            mod.impl.logInf("%s: GGUF version: %d\n", .{ func, @as(c_int, @intCast(mod.impl.c.gguf_get_version(self.ctx_gguf))) });
            mod.impl.logInf("%s: alignment:    %zu\n", .{ func, mod.impl.c.gguf_get_alignment(self.ctx_gguf) });
            mod.impl.logInf("%s: n_tensors:    %d\n", .{ func, n_tensors });
            mod.impl.logInf("%s: n_kv:         %d\n", .{ func, @as(c_int, @intCast(mod.impl.c.gguf_get_n_kv(self.ctx_gguf))) });
            mod.impl.logInf("\n", .{});
        }

        // modalities
        {
            try self.getBool(mod.names.KEY_HAS_VISION_ENC, &self.has_vision, false);
            try self.getBool(mod.names.KEY_HAS_AUDIO_ENC, &self.has_audio, false);
            try self.getBool(mod.names.KEY_HAS_GEN_AUDIO_ENC, &self.has_gen_audio, false);

            if (self.has_vision) mod.impl.logInf("%s: has vision encoder\n", .{func});
            if (self.has_audio) mod.impl.logInf("%s: has audio encoder\n", .{func});
            if (self.has_gen_audio) mod.impl.logInf("%s: has audio generation (gen) encoder\n", .{func});
        }

        // tensors
        if (!skip_tensors) {
            var i: c_int = 0;
            while (i < n_tensors) : (i += 1) {
                const tname = mod.impl.c.gguf_get_tensor_name(self.ctx_gguf, i);
                const offset = mod.impl.c.gguf_get_tensor_offset(self.ctx_gguf, i);
                const ttype = mod.impl.c.gguf_get_tensor_type(self.ctx_gguf, i);
                const cur = mod.impl.c.ggml_get_tensor(meta, tname);
                const tensor_size = mod.impl.c.ggml_nbytes(cur);
                self.model_size += tensor_size;
                mod.impl.logDbg("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%llu, %llu, %llu, %llu], type = %s\n", .{
                    func,                                    i,                                       mod.impl.c.ggml_n_dims(cur),             @as([*:0]const u8, @ptrCast(&cur.*.name)),
                    tensor_size,                             offset,                                  @as(c_ulonglong, @bitCast(cur.*.ne[0])), @as(c_ulonglong, @bitCast(cur.*.ne[1])),
                    @as(c_ulonglong, @bitCast(cur.*.ne[2])), @as(c_ulonglong, @bitCast(cur.*.ne[3])), mod.impl.c.ggml_type_name(ttype),
                });
            }
        }
        return self;
    }

    /// The smart pointers' destructors.
    pub fn deinit(self: *Loader) void {
        if (self.ctx_gguf != null) mod.impl.c.gguf_free(self.ctx_gguf);
        if (self.ctx_meta != null) mod.impl.c.ggml_free(self.ctx_meta);
    }

    fn findKey(self: *const Loader, key: [*:0]const u8) i64 {
        return mod.impl.c.gguf_find_key(self.ctx_gguf, key);
    }

    /// Ports `get_bool` (mtmd-model.cpp:3572 @b1b770d90), a member of `mtmd_model_loader`.
    pub fn getBool(self: *const Loader, key: [*:0]const u8, output: *bool, required: bool) Error!void {
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        output.* = mod.impl.c.gguf_get_val_bool(self.ctx_gguf, i);
    }

    /// Ports `get_i32` (mtmd-model.cpp:3583 @b1b770d90), a member of `mtmd_model_loader`.
    pub fn getI32(self: *const Loader, key: [*:0]const u8, output: *i32, required: bool) Error!void {
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        output.* = mod.impl.c.gguf_get_val_i32(self.ctx_gguf, i);
    }

    /// Ports `get_u32` (mtmd-model.cpp:3594 @b1b770d90), a member of `mtmd_model_loader`.
    pub fn getU32(self: *const Loader, key: [*:0]const u8, output: *i32, required: bool) Error!void {
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        const val = mod.impl.c.gguf_get_val_u32(self.ctx_gguf, i);
        // sanity check
        if (val > @as(u32, std.math.maxInt(i32))) {
            return throwErr("%s: value %u for key '%s' exceeds INT32_MAX\n", .{ "get_u32", val, key });
        }
        output.* = @intCast(val);
    }

    /// Ports `get_f32` (mtmd-model.cpp:3611 @b1b770d90), a member of `mtmd_model_loader`.
    pub fn getF32(self: *const Loader, key: [*:0]const u8, output: *f32, required: bool) Error!void {
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        output.* = mod.impl.c.gguf_get_val_f32(self.ctx_gguf, i);
    }

    /// Ports `get_arr_f32` (mtmd-model.cpp:3622 @b1b770d90), a member of `mtmd_model_loader`.
    pub fn getArrF32(self: *const Loader, key: [*:0]const u8, output: *std.ArrayList(f32), required: bool) Error!void {
        const func = "get_arr_f32";
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        if (mod.impl.c.gguf_get_kv_type(self.ctx_gguf, i) != mod.impl.c.GGUF_TYPE_ARRAY) return throwErr("%s: key '%s' is not an array\n", .{ func, key });
        const t = mod.impl.c.gguf_get_arr_type(self.ctx_gguf, i);
        if (t != mod.impl.c.GGUF_TYPE_FLOAT32) return throwErr("%s: array '%s' has type %d, expected %d (GGUF_TYPE_FLOAT32)\n", .{ func, key, @as(c_int, @intCast(t)), @as(c_int, mod.impl.c.GGUF_TYPE_FLOAT32) });
        const n = mod.impl.c.gguf_get_arr_n(self.ctx_gguf, i);
        if (n > @as(usize, std.math.maxInt(c_int))) return throwErr("%s: array '%s' is too large (%zu elements)\n", .{ func, key, n });
        try output.resize(std.heap.c_allocator, n);
        const values: [*]align(1) const f32 = @ptrCast(mod.impl.c.gguf_get_arr_data(self.ctx_gguf, i));
        for (0..n) |j| output.items[j] = values[j];
    }

    /// Ports `get_string` (mtmd-model.cpp:3648 @b1b770d90), a member of `mtmd_model_loader`:
    /// the value, borrowed from the GGUF context, or `output` left alone.
    pub fn getStringPtr(self: *const Loader, key: [*:0]const u8, output: *?[*:0]const u8, required: bool) Error!void {
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        output.* = mod.impl.c.gguf_get_val_str(self.ctx_gguf, i);
    }

    /// `get_string` into an owned copy, the C++'s `std::string` output.
    /// The copy keeps a NUL past its end, so `c_str()` is `output.ptr`.
    pub fn getString(self: *const Loader, key: [*:0]const u8, output: *[]const u8, required: bool) Error!void {
        var p: ?[*:0]const u8 = null;
        try self.getStringPtr(key, &p, required);
        if (p) |s| output.* = try std.heap.c_allocator.dupeSentinel(u8, std.mem.span(s), 0);
    }

    /// Ports `get_arr_int` (mtmd-model.cpp:3659 @b1b770d90), a member of `mtmd_model_loader`.
    pub fn getArrInt(self: *const Loader, key: [*:0]const u8, output: *std.ArrayList(i32), required: bool) Error!void {
        const func = "get_arr_int";
        const i = self.findKey(key);
        if (i < 0) {
            if (required) return keyNotFound(key);
            return;
        }
        if (mod.impl.c.gguf_get_kv_type(self.ctx_gguf, i) != mod.impl.c.GGUF_TYPE_ARRAY) return throwErr("%s: key '%s' is not an array\n", .{ func, key });
        const t = mod.impl.c.gguf_get_arr_type(self.ctx_gguf, i);
        if (t != mod.impl.c.GGUF_TYPE_INT32) return throwErr("%s: array '%s' has type %d, expected %d (GGUF_TYPE_INT32)\n", .{ func, key, @as(c_int, @intCast(t)), @as(c_int, mod.impl.c.GGUF_TYPE_INT32) });
        const n = mod.impl.c.gguf_get_arr_n(self.ctx_gguf, i);
        if (n > @as(usize, std.math.maxInt(c_int))) return throwErr("%s: array '%s' is too large (%zu elements)\n", .{ func, key, n });
        try output.resize(std.heap.c_allocator, n);
        const values: [*]align(1) const i32 = @ptrCast(mod.impl.c.gguf_get_arr_data(self.ctx_gguf, i));
        for (0..n) |j| output.items[j] = values[j];
    }

    fn keyNotFound(key: [*:0]const u8) Error {
        return throwErr("Key not found: %s", .{key});
    }

    /// Ports `set_llava_uhd_res_candidates` (mtmd-model.cpp:3685 @b1b770d90), a member of `mtmd_model_loader`.
    pub fn setLlavaUhdResCandidates(model: *mod.model.Model, max_patches_per_side: c_int) Error!void {
        const hparams = &model.hparams;
        var x: c_int = 1;
        while (x <= max_patches_per_side) : (x += 1) {
            var y: c_int = 1;
            while (y <= max_patches_per_side) : (y += 1) {
                if (x == 1 and y == 1) continue; // skip the first point
                try hparams.image_res_candidates.append(std.heap.c_allocator, .{ .width = x * hparams.image_size, .height = y * hparams.image_size });
            }
        }
    }

    /// Ports `set_internvl_dhr_res_candidates` (mtmd-model.cpp:3700 @b1b770d90), a member of `mtmd_model_loader`.
    pub fn setInternvlDhrResCandidates(model: *mod.model.Model) Error!void {
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
                try hparams.image_res_candidates.append(std.heap.c_allocator, .{ .width = a * hparams.image_size, .height = b * hparams.image_size });
            }
        }
    }

    /// Ports `load_hparams` (mtmd-model.cpp:117 @b1b770d90), a member of `mtmd_model_loader`;
    /// the model-specific switch is `arms_hparams.load`.
    pub fn loadHparams(self: *Loader, model: *mod.model.Model, modality: mod.impl.Modality) Error!void {
        const func = "load_hparams";
        const hparams = &model.hparams;
        var log_ffn_op: [*:0]const u8 = ""; // for logging

        // sanity check
        switch (modality) {
            .vision => mod.impl.ggmlAssert(self.has_vision, "mtmd-model.cpp", 123, "has_vision"),
            .audio => mod.impl.ggmlAssert(self.has_audio, "mtmd-model.cpp", 125, "has_audio"),
            .gen_audio => mod.impl.ggmlAssert(self.has_gen_audio, "mtmd-model.cpp", 127, "has_gen_audio"),
        }
        model.modality = modality;

        // projector type
        var proj_type: ?[*:0]const u8 = null;
        {
            // default key
            try self.getStringPtr(mod.names.KEY_PROJ_TYPE, &proj_type, false);

            // for models with mixed modalities
            if (proj_type == null or proj_type.?[0] == 0) {
                switch (modality) {
                    .vision => try self.getStringPtr(mod.names.KEY_VISION_PROJ_TYPE, &proj_type, false),
                    .audio => try self.getStringPtr(mod.names.KEY_AUDIO_PROJ_TYPE, &proj_type, false),
                    .gen_audio => try self.getStringPtr(mod.names.KEY_GEN_AUDIO_PROJ_TYPE, &proj_type, false),
                }
            }
            if (proj_type == null) proj_type = "";

            model.proj_type = mod.names.projectorTypeFromString(std.mem.span(proj_type.?));

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
            try self.getU32(fmtKey(&kb, mod.names.KEY_N_EMBD, prefix), &hparams.n_embd, true);
            try self.getU32(fmtKey(&kb, mod.names.KEY_N_HEAD, prefix), &hparams.n_head, true);
            try self.getU32(fmtKey(&kb, mod.names.KEY_N_EMBD_HEAD, prefix), &hparams.n_embd_head, false);
            try self.getU32(fmtKey(&kb, mod.names.KEY_N_FF, prefix), &hparams.n_ff, true);
            try self.getU32(fmtKey(&kb, mod.names.KEY_N_BLOCK, prefix), &hparams.n_layer, true);
            try self.getU32(fmtKey(&kb, mod.names.KEY_PROJ_DIM, prefix), &hparams.projection_dim, true);
            try self.getF32(fmtKey(&kb, mod.names.KEY_LAYER_NORM_EPS, prefix), &hparams.eps, true);

            // n_head_kv is optional (for GQA), default to n_head
            hparams.n_head_kv = hparams.n_head;
            try self.getU32(fmtKey(&kb, mod.names.KEY_N_HEAD_KV, prefix), &hparams.n_head_kv, false);

            if (is_vision) {
                try self.getU32(mod.names.KEY_IMAGE_SIZE, &hparams.image_size, true);
                try self.getU32(mod.names.KEY_PATCH_SIZE, &hparams.patch_size, true);
                try self.getU32(mod.names.KEY_MAX_SLICE_NUMS, &hparams.max_slice_nums, false);
                try self.getI32(mod.names.KEY_MINICPMV_VERSION, &hparams.minicpmv_version, false); // legacy
                try self.getU32(mod.names.KEY_MINICPMV_QUERY_NUM, &hparams.minicpmv_query_num, false);
                if (hparams.minicpmv_query_num == 0) {
                    // Fallback to hardcoded values for legacy models
                    hparams.minicpmv_query_num = switch (hparams.minicpmv_version) {
                        3, 4, 5, 6, 100045 => 64,
                        else => 96,
                    };
                }
            } else if (is_audio) {
                try self.getU32(mod.names.KEY_A_NUM_MEL_BINS, &hparams.n_mel_bins, true);
                // some hparams are unused, but still need to set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;
            } else if (is_gen_audio) {
                // these are unused, but still need to be set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;
                try self.getString(mod.names.KEY_GEN_AUDIO_VARIANT, &hparams.gen_model_variant, false);
            } else {
                mod.impl.ggmlAssert(false, "mtmd-model.cpp", 219, "false && \"unknown modality\"");
            }

            // for pinpoints, we need to convert it into a list of resolution candidates
            {
                var pinpoints: std.ArrayList(i32) = .empty;
                defer pinpoints.deinit(std.heap.c_allocator);
                try self.getArrInt(mod.names.KEY_IMAGE_GRID_PINPOINTS, &pinpoints, false);
                if (pinpoints.items.len % 2 != 0) {
                    return throwErr("%s: image_grid_pinpoints must have an even number of elements, got %zu\n", .{ func, pinpoints.items.len });
                }
                var i: usize = 0;
                while (i < pinpoints.items.len) : (i += 2) {
                    try hparams.image_res_candidates.append(std.heap.c_allocator, .{ .width = pinpoints.items[i], .height = pinpoints.items[i + 1] });
                }
            }

            // default warmup value
            hparams.warmup_image_size = hparams.image_size;

            {
                var use_gelu = false;
                var use_silu = false;
                try self.getBool(mod.names.KEY_USE_GELU, &use_gelu, false);
                try self.getBool(mod.names.KEY_USE_SILU, &use_silu, false);
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
                try self.getStringPtr(mod.names.KEY_MM_PATCH_MERGE_TYPE, &mm_patch_merge_type, false);
                if (mm_patch_merge_type != null and std.mem.eql(u8, std.mem.span(mm_patch_merge_type.?), "spatial_unpad")) {
                    hparams.mm_patch_merge_type = .spatial_unpad;
                }
            }

            if (is_vision) {
                var image_mean: std.ArrayList(f32) = .empty;
                defer image_mean.deinit(std.heap.c_allocator);
                var image_std: std.ArrayList(f32) = .empty;
                defer image_std.deinit(std.heap.c_allocator);
                try self.getArrF32(mod.names.KEY_IMAGE_MEAN, &image_mean, false);
                try self.getArrF32(mod.names.KEY_IMAGE_STD, &image_std, false);
                if (image_mean.items.len < 3 or image_std.items.len < 3) {
                    return throwErr("%s: image_mean/image_std arrays must have at least 3 elements, got %zu and %zu\n", .{ func, image_mean.items.len, image_std.items.len });
                }
                for (0..3) |i| {
                    hparams.image_mean[i] = image_mean.items[i];
                    hparams.image_std[i] = image_std.items[i];
                }
            }

            // Load the vision/audio feature layer indices if they are explicitly provided
            try self.getArrInt(fmtKey(&kb, mod.names.KEY_FEATURE_LAYERS, prefix), &hparams.feature_layers, false);
            for (hparams.feature_layers.items) |v| {
                if (v > hparams.n_layer) {
                    return throwErr("%s: feature layer index %d is out of range (n_layer: %d)", .{ func, @as(c_int, v), @as(c_int, hparams.n_layer) });
                }
            }

            // model-specific params
            try mod.arms_hparams.load(self, model, .{
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

            mod.impl.logInf("%s: projector:          %s\n", .{ func, proj_type.? });
            mod.impl.logInf("%s: n_embd:             %d\n", .{ func, hparams.n_embd });
            mod.impl.logInf("%s: n_head:             %d\n", .{ func, hparams.n_head });
            mod.impl.logInf("%s: n_ff:               %d\n", .{ func, hparams.n_ff });
            mod.impl.logInf("%s: n_layer:            %d\n", .{ func, hparams.n_layer });
            mod.impl.logInf("%s: ffn_op:             %s\n", .{ func, log_ffn_op });
            mod.impl.logInf("%s: projection_dim:     %d\n", .{ func, hparams.projection_dim });
            if (is_vision) {
                mod.impl.logInf("\n--- vision hparams ---\n", .{});
                mod.impl.logInf("%s: image_size:         %d\n", .{ func, hparams.image_size });
                mod.impl.logInf("%s: patch_size:         %d\n", .{ func, hparams.patch_size });
                mod.impl.logInf("%s: has_llava_proj:     %d\n", .{ func, @as(c_int, @intFromBool(hparams.has_llava_projector)) });
                mod.impl.logInf("%s: minicpmv_version:   %d\n", .{ func, hparams.minicpmv_version });
                mod.impl.logInf("%s: n_merge:            %d\n", .{ func, hparams.n_merge });
                mod.impl.logInf("%s: n_wa_pattern: %d\n", .{ func, hparams.n_wa_pattern });
                if (hparams.wa_layer_indexes.count() != 0) {
                    mod.impl.logInf("%s: wa_layer_indexes:  ", .{func});
                    for (try mod.arms_hparams.waLayerIndexesInSetOrder(hparams)) |layer| mod.impl.logInf("%d ", .{layer});
                    mod.impl.logInf("\n", .{});
                }
                if (hparams.image_min_pixels > 0) {
                    mod.impl.logInf("%s: image_min_pixels:   %d%s\n", .{ func, hparams.image_min_pixels, @as([*:0]const u8, if (hparams.custom_image_min_tokens > 0) " (custom value)" else "") });
                }
                if (hparams.image_max_pixels > 0) {
                    mod.impl.logInf("%s: image_max_pixels:   %d%s\n", .{ func, hparams.image_max_pixels, @as([*:0]const u8, if (hparams.custom_image_max_tokens > 0) " (custom value)" else "") });
                }
                if (hparams.preproc_max_tiles > 0) {
                    mod.impl.logInf("%s: preproc_tiles:      %d - %d\n", .{ func, hparams.preproc_min_tiles, hparams.preproc_max_tiles });
                }
            } else if (is_audio) {
                mod.impl.logInf("\n--- audio hparams ---\n", .{});
                mod.impl.logInf("%s: n_mel_bins:         %d\n", .{ func, hparams.n_mel_bins });
                mod.impl.logInf("%s: proj_stack_factor:  %d\n", .{ func, hparams.proj_stack_factor });
                mod.impl.logInf("%s: audio_chunk_len:    %d\n", .{ func, hparams.audio_chunk_len });
                mod.impl.logInf("%s: audio_sample_rate:  %d\n", .{ func, hparams.audio_sample_rate });
                mod.impl.logInf("%s: audio_n_fft:        %d\n", .{ func, hparams.audio_n_fft });
                mod.impl.logInf("%s: audio_window_len:   %d\n", .{ func, hparams.audio_window_len });
                mod.impl.logInf("%s: audio_hop_len:      %d\n", .{ func, hparams.audio_hop_len });

                // GEMMA4UA and pocket-tts are encoder-free: no FFT/filterbank
                // Muse Spark is encoder-free in the same sense: its conv pyramid reads the raw waveform, one channel takes the place of the mel bins
                const fft_based = model.proj_type != .gemma4ua and model.proj_type != .muse_spark_audio and model.proj_type != .pockettts_spkenc;

                // Validate audio hparams loaded from GGUF metadata
                if (hparams.n_mel_bins <= 0 or (fft_based and hparams.n_mel_bins > 256)) {
                    return throwErr("%s: n_mel_bins (%d) must be in range [1, 256]\n", .{ func, hparams.n_mel_bins });
                }
                if (fft_based and (hparams.audio_sample_rate <= 0 or hparams.audio_n_fft <= 0 or hparams.audio_hop_len <= 0 or hparams.audio_window_len <= 0)) {
                    return throwErr("%s: audio hparams invalid: sample_rate=%d n_fft=%d window_len=%d hop_len=%d\n", .{ func, hparams.audio_sample_rate, hparams.audio_n_fft, hparams.audio_window_len, hparams.audio_hop_len });
                }
            }
            mod.impl.logInf("\n", .{});
            mod.impl.logInf("%s: model size:         %.2f MiB\n", .{ func, @as(f64, @floatFromInt(self.model_size)) / 1024.0 / 1024.0 });
            mod.impl.logInf("%s: metadata size:      %.2f MiB\n", .{ func, @as(f64, @floatFromInt(mod.impl.c.ggml_get_mem_size(self.ctx_meta))) / 1024.0 / 1024.0 });
        }
    }

    /// Ports `load_tensors` (mtmd-model.cpp:1321 @b1b770d90), a member of `mtmd_model_loader`;
    /// the model-specific switch is `arms_tensors.load`.
    pub fn loadTensors(self: *Loader, model: *mod.model.Model, buft: mod.impl.c.ggml_backend_buffer_type_t, no_alloc: bool) Error!void {
        const func = "load_tensors";
        const hparams = &model.hparams;

        var tl: TensorLoader = .{ .loader = self, .model = model, .buft = buft };
        defer tl.deinit();

        tl.fin = std.c.fopen(self.fname.ptr, "rb");
        if (tl.fin == null) return throwErr("%s: failed to open %s\n", .{ func, self.fname.ptr });

        const prefix: [*:0]const u8 = if (model.modality == .audio) "a" else if (model.modality == .gen_audio) "a.gen.code" else "v";
        tl.prefix = prefix;

        // get offsets
        {
            var i: i64 = 0;
            while (i < mod.impl.c.gguf_get_n_tensors(self.ctx_gguf)) : (i += 1) {
                const tname = std.mem.span(mod.impl.c.gguf_get_tensor_name(self.ctx_gguf, i));
                try tl.tensor_offset.put(std.heap.c_allocator, tname, mod.impl.c.gguf_get_data_offset(self.ctx_gguf) + mod.impl.c.gguf_get_tensor_offset(self.ctx_gguf, i));
            }
        }

        // create data context
        const params: mod.impl.c.ggml_init_params = .{
            .mem_size = @as(usize, @intCast(mod.impl.c.gguf_get_n_tensors(self.ctx_gguf) + 1)) * mod.impl.c.ggml_tensor_overhead(),
            .mem_buffer = null,
            .no_alloc = true,
        };
        model.ctx_data = mod.impl.c.ggml_init(params);
        if (model.ctx_data == null) return throwErr("%s: failed to init ggml context\n", .{func});

        var nb: [256]u8 = undefined;
        model.class_embedding = try tl.getTensor(mod.names.TN_CLASS_EMBD, false);

        model.pre_ln_w = try tl.getTensor(fmtKey2(&nb, mod.names.TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = try tl.getTensor(fmtKey2(&nb, mod.names.TN_LN_PRE, prefix, "bias"), false);

        model.post_ln_w = try tl.getTensor(fmtKey2(&nb, mod.names.TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = try tl.getTensor(fmtKey2(&nb, mod.names.TN_LN_POST, prefix, "bias"), false);

        // every vision encoder starts with a patch embedding, except the ones with their own stem
        const has_patch_embd = (model.modality == .vision and
            model.proj_type != .gemma3nv and
            model.proj_type != .deepseekocr and
            model.proj_type != .deepseekocr2);

        model.patch_bias = try tl.getTensor(mod.names.TN_PATCH_BIAS, false);
        model.patch_embeddings_0 = try tl.getTensor(mod.names.TN_PATCH_EMBD, has_patch_embd);
        model.patch_embeddings_1 = try tl.getTensor(mod.names.TN_PATCH_EMBD_1, false);

        model.norm_embd_w = try tl.getTensor(fmtKey(&nb, mod.names.TN_NORM_EMBD, "weight"), false);
        model.norm_embd_b = try tl.getTensor(fmtKey(&nb, mod.names.TN_NORM_EMBD, "bias"), false);

        model.position_embeddings = try tl.getTensor(fmtKey(&nb, mod.names.TN_POS_EMBD, prefix), false);

        // lfm3vl
        model.lfm3vl_embedding_norm_w = try tl.getTensor(fmtKey2(&nb, mod.names.TN_LFM3VL_EMBEDDING_NORM, prefix, "weight"), false);

        const has_standard_layers = model.proj_type != .gemma3nv and model.proj_type != .qwen3tts_spkenc and model.proj_type != .pockettts_gen and model.proj_type != .soprano and model.proj_type != .muse_spark_audio and
            model.proj_type != .chatterbox and model.proj_type != .chatterbox_spkenc;

        // layers
        const n_layers_to_load: usize = if (has_standard_layers) @intCast(hparams.n_layer) else 0;
        try model.layers.resize(std.heap.c_allocator, n_layers_to_load);
        for (model.layers.items) |*l| l.* = .{};
        for (0..n_layers_to_load) |ilu| {
            const il: c_int = @intCast(ilu);
            const layer = &model.layers.items[ilu];
            layer.k_w = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_K, prefix, il, "weight"), false);
            layer.q_w = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_Q, prefix, il, "weight"), false);
            layer.v_w = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_V, prefix, il, "weight"), false);
            // lfm3vl / lfm3a conv layers have no attention tensors
            const o_w_required = model.proj_type != .lfm3vl and model.proj_type != .lfm3a;
            layer.o_w = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_OUTPUT, prefix, il, "weight"), o_w_required);
            layer.qkv_w = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_QKV, prefix, il, "weight"), false);
            layer.k_norm = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_K_NORM, prefix, il, "weight"), false);
            layer.q_norm = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_Q_NORM, prefix, il, "weight"), false);
            layer.ln_1_w = try tl.getTensor(fmtL(&nb, mod.names.TN_LN_1, prefix, il, "weight"), false);
            layer.ln_2_w = try tl.getTensor(fmtL(&nb, mod.names.TN_LN_2, prefix, il, "weight"), false);
            layer.ls_1_w = try tl.getTensor(fmtL(&nb, mod.names.TN_LS_1, prefix, il, "weight"), false); // no bias
            layer.ls_2_w = try tl.getTensor(fmtL(&nb, mod.names.TN_LS_2, prefix, il, "weight"), false); // no bias
            layer.ls_out_w = try tl.getTensor(fmtL(&nb, mod.names.TN_LS_OUT, prefix, il, "weight"), false); // no bias
            layer.attn_post_norm_w = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_POST_NORM, prefix, il, "weight"), false); // no bias
            layer.ff_post_norm_w = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_POST_NORM, prefix, il, "weight"), false); // no bias

            layer.k_b = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_K, prefix, il, "bias"), false);
            layer.q_b = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_Q, prefix, il, "bias"), false);
            layer.v_b = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_V, prefix, il, "bias"), false);
            layer.o_b = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_OUTPUT, prefix, il, "bias"), false);
            layer.qkv_b = try tl.getTensor(fmtL(&nb, mod.names.TN_ATTN_QKV, prefix, il, "bias"), false);
            layer.ln_1_b = try tl.getTensor(fmtL(&nb, mod.names.TN_LN_1, prefix, il, "bias"), false);
            layer.ln_2_b = try tl.getTensor(fmtL(&nb, mod.names.TN_LN_2, prefix, il, "bias"), false);

            // MoE ffn (dots3note vision pyramid blocks); replaces the dense ffn when present
            layer.ff_gate_inp_w = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_GATE_INP, prefix, il, "weight"), false);
            layer.ff_gate_exps_w = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_GATE_EXPS, prefix, il, "weight"), false);
            layer.ff_up_exps_w = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_UP_EXPS, prefix, il, "weight"), false);
            layer.ff_down_exps_w = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_DOWN_EXPS, prefix, il, "weight"), false);
            layer.ff_exp_probs_b = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_EXP_PROBS_B, prefix, il, "weight"), false);
            const is_moe = layer.ff_gate_exps_w != null;

            // ffn
            layer.ff_up_w = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_UP, prefix, il, "weight"), !is_moe);
            layer.ff_up_b = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_UP, prefix, il, "bias"), false);
            layer.ff_gate_w = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_GATE, prefix, il, "weight"), false);
            layer.ff_gate_b = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_GATE, prefix, il, "bias"), false);
            layer.ff_down_w = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_DOWN, prefix, il, "weight"), !is_moe);
            layer.ff_down_b = try tl.getTensor(fmtL(&nb, mod.names.TN_FFN_DOWN, prefix, il, "bias"), false);

            // mimovl per-head attention sink bias
            layer.attn_sinks = try tl.getTensor(fmtPI(&nb, mod.names.TN_ATTN_SINKS, prefix, il), false);

            // qwen3vl deepstack layer
            layer.deepstack_norm_w = try tl.getTensor(fmtIS(&nb, mod.names.TN_DEEPSTACK_NORM, il, "weight"), false);
            layer.deepstack_norm_b = try tl.getTensor(fmtIS(&nb, mod.names.TN_DEEPSTACK_NORM, il, "bias"), false);
            layer.deepstack_fc1_w = try tl.getTensor(fmtIS(&nb, mod.names.TN_DEEPSTACK_FC1, il, "weight"), false);
            layer.deepstack_fc1_b = try tl.getTensor(fmtIS(&nb, mod.names.TN_DEEPSTACK_FC1, il, "bias"), false);
            layer.deepstack_fc2_w = try tl.getTensor(fmtIS(&nb, mod.names.TN_DEEPSTACK_FC2, il, "weight"), false);
            layer.deepstack_fc2_b = try tl.getTensor(fmtIS(&nb, mod.names.TN_DEEPSTACK_FC2, il, "bias"), false);
            if (layer.hasDeepstack()) model.n_deepstack_layers += 1;

            // lfm3vl conv layers
            layer.conv2d_in_proj_w = try tl.getTensor(fmtL(&nb, mod.names.TN_LFM3VL_CONV2D_IN_PROJ, prefix, il, "weight"), false);
            layer.conv2d_spatial_w = try tl.getTensor(fmtL(&nb, mod.names.TN_LFM3VL_CONV2D_SPATIAL, prefix, il, "weight"), false);
            layer.conv2d_out_proj_w = try tl.getTensor(fmtL(&nb, mod.names.TN_LFM3VL_CONV2D_OUT_PROJ, prefix, il, "weight"), false);

            // some models already exported with legacy (incorrect) naming which is quite messy, let's fix it here
            // note: Qwen model converted from the old surgery script has n_ff = 0, so we cannot use n_ff to check!
            const pt = model.proj_type;
            const is_ffn_swapped = (pt == .mlp or pt == .mlp_norm or pt == .ldp or pt == .ldpv2 or pt == .qwen2vl or pt == .qwen25vl or
                pt == .exaone4_5 or pt == .glm_edge or pt == .gemma3 or pt == .idefics3 or pt == .minicpmv or pt == .minicpmv4_6 or pt == .minicpmv4_7 or pt == .minicpm_ocr_1_0) and
                layer.ff_up_w != null and layer.ff_down_w != null and mod.graph.ne(layer.ff_down_w, 0) == hparams.n_embd;
            if (is_ffn_swapped) {
                // swap up and down weights
                std.mem.swap(mod.graph.T, &layer.ff_up_w, &layer.ff_down_w);
                // swap up and down biases
                std.mem.swap(mod.graph.T, &layer.ff_up_b, &layer.ff_down_b);
                if (il == 0) mod.impl.logWrn("%s: ffn up/down are swapped\n", .{func});
            }
        }

        // model-specific tensors
        try mod.arms_tensors.load(&tl);

        // load data
        {
            var read_buf: std.ArrayList(u8) = .empty;
            defer read_buf.deinit(std.heap.c_allocator);

            // start loading event
            if (self.progress_callback) |cb| _ = cb(0.0, self.progress_callback_user_data);

            // compute total tensor data size for progress reporting
            var total_data_size: usize = 0;
            for (tl.tensors_to_load.items) |t| total_data_size += mod.impl.c.ggml_nbytes(t);

            // alloc memory and offload data
            model.buf = mod.impl.c.ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_data, buft);
            mod.impl.c.ggml_backend_buffer_set_usage(model.buf, mod.impl.c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // read the weight from file
            if (!no_alloc) {
                var data_loaded: usize = 0;
                for (tl.tensors_to_load.items) |t| {
                    const tname: [*:0]const u8 = @ptrCast(&t.?.name);
                    const cur = mod.impl.c.ggml_get_tensor(model.ctx_data, tname);
                    mod.impl.ggmlAssert(cur != null, "mtmd-model.cpp", 3537, "cur && \"tensor not found in ctx_data\"");
                    const off = tl.tensor_offset.get(std.mem.span(tname));
                    mod.impl.ggmlAssert(off != null, "mtmd-model.cpp", 3539, "it_off != tensor_offset.end() && \"no offset for tensor\"");
                    if (stdlib.libc.fseeko(tl.fin, @intCast(off.?), 0) != 0) {
                        return throwErr("%s: failed to seek for tensor %s\n", .{ func, tname });
                    }
                    const num_bytes = mod.impl.c.ggml_nbytes(cur);
                    if (mod.impl.c.ggml_backend_buft_is_host(buft)) {
                        // for the CPU and Metal backend, we can read directly into the tensor
                        _ = std.c.fread(@ptrCast(cur.*.data), 1, num_bytes, tl.fin.?);
                    } else {
                        // read into a temporary buffer first, then copy to device memory
                        try read_buf.resize(std.heap.c_allocator, num_bytes);
                        _ = std.c.fread(read_buf.items.ptr, 1, num_bytes, tl.fin.?);
                        mod.impl.c.ggml_backend_tensor_set(cur, read_buf.items.ptr, 0, num_bytes);
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
                mod.impl.logDbg("%s: loaded %zu tensors from %s\n", .{ func, tl.tensors_to_load.items.len, self.fname.ptr });
            } else {
                mod.impl.logDbg("%s: no_alloc is set, skipping tensor data loading (%zu tensors)\n", .{ func, tl.tensors_to_load.items.len });
            }
        }
    }
};

/// `load_tensors`'s locals and lambdas, handed to `arms_tensors.load`.
pub const TensorLoader = struct {
    loader: *Loader,
    model: *mod.model.Model,
    buft: mod.impl.c.ggml_backend_buffer_type_t,
    prefix: [*:0]const u8 = "v",
    fin: ?*std.c.FILE = null,
    tensor_offset: std.StringHashMapUnmanaged(usize) = .empty,
    tensors_to_load: std.ArrayList(mod.graph.T) = .empty,
    loaded_tensor_names: std.StringHashMapUnmanaged(void) = .empty,

    fn deinit(self: *TensorLoader) void {
        if (self.fin) |f| _ = std.c.fclose(f);
        self.tensor_offset.deinit(std.heap.c_allocator);
        self.tensors_to_load.deinit(std.heap.c_allocator);
        var it = self.loaded_tensor_names.keyIterator();
        while (it.next()) |k| std.heap.c_allocator.free(k.*);
        self.loaded_tensor_names.deinit(std.heap.c_allocator);
    }

    /// The `get_tensor` lambda (mtmd-model.cpp:1355 @b1b770d90).
    pub fn getTensor(self: *TensorLoader, name: [*:0]const u8, required: bool) Error!mod.graph.T {
        // `__func__` inside a lambda is the closure's call operator's name
        const func = "operator()";
        const key = std.mem.span(name);
        // Each tensor should only be loaded once; duplicates indicate a bug
        if (self.loaded_tensor_names.contains(key)) return throwErr("%s: tensor already loaded: %s\n", .{ func, name });
        var cur: mod.graph.T = mod.impl.c.ggml_get_tensor(self.loader.ctx_meta, name);
        if (cur == null and required) return throwErr("%s: unable to find tensor %s\n", .{ func, name });
        if (cur != null) {
            try self.tensors_to_load.append(std.heap.c_allocator, cur);
            const data_tensor = mod.impl.c.ggml_dup_tensor(self.model.ctx_data, cur);
            _ = mod.impl.c.ggml_set_name(data_tensor, @ptrCast(&cur.?.name));
            try self.loaded_tensor_names.put(std.heap.c_allocator, try std.heap.c_allocator.dupe(u8, key), {});
            cur = data_tensor;
            // add to weight memory counter
            try self.model.mem_weights.add(mod.impl.c.ggml_backend_buft_get_device(self.buft), mod.impl.c.ggml_nbytes(cur));
        }
        return cur;
    }

    /// The `get_vector` lambda (mtmd-model.cpp:1401 @b1b770d90). The caller owns
    /// the list.
    pub fn getVector(self: *TensorLoader, name: [*:0]const u8) Error!std.ArrayList(f32) {
        const func = "load_tensors";
        var result: std.ArrayList(f32) = .empty;
        const off = self.tensor_offset.get(std.mem.span(name)) orelse return result;

        const idx = mod.impl.c.gguf_find_tensor(self.loader.ctx_gguf, name);
        if (idx < 0) return throwErr("%s: failed to find tensor %s\n", .{ func, name });

        const t = mod.impl.c.gguf_get_tensor_type(self.loader.ctx_gguf, idx);
        if (t != mod.impl.c.GGML_TYPE_F32) return throwErr("%s: %s must be %s, was %s\n", .{ func, name, mod.impl.c.ggml_type_name(mod.impl.c.GGML_TYPE_F32), mod.impl.c.ggml_type_name(t) });

        const n_bytes = mod.impl.c.gguf_get_tensor_size(self.loader.ctx_gguf, idx);
        if (n_bytes == 0) return throwErr("%s: tensor %s is empty\n", .{ func, name });

        const n_elems = n_bytes / @sizeOf(f32);
        try result.resize(std.heap.c_allocator, n_elems);
        _ = stdlib.libc.fseeko(self.fin, @intCast(off), 0);
        _ = std.c.fread(@ptrCast(result.items.ptr), 1, n_bytes, self.fin.?);
        return result;
    }

    /// `fin.seekg(tensor_offset.at(name), std::ios::beg); fin.read(dst, n)`,
    /// as `load_tensors` reads a voice tensor: false if the read came up short. `at` throwing
    /// `std::out_of_range` cannot happen for a tensor the file lists.
    pub fn readTensorBytes(self: *TensorLoader, name: []const u8, dst: []u8) bool {
        const off = self.tensor_offset.get(name) orelse mod.impl.abortUncaught("std::out_of_range", "unordered_map::at: key not found");
        if (stdlib.libc.fseeko(self.fin, @intCast(off), 0) != 0) return false;
        return std.c.fread(dst.ptr, 1, dst.len, self.fin.?) == dst.len;
    }

    /// The `get_scalar` lambda (mtmd-model.cpp:1430 @b1b770d90).
    pub fn getScalar(self: *TensorLoader, name: [*:0]const u8, default_val: f32) Error!f32 {
        var v = try self.getVector(name);
        defer v.deinit(std.heap.c_allocator);
        if (v.items.len == 0) return default_val;
        if (v.items.len != 1) return throwErr("%s: expected scalar tensor '%s' but got %d elements\n", .{ "load_tensors", name, @as(c_int, @intCast(v.items.len)) });
        return v.items[0];
    }

    /// The `load_seanet` lambda (mtmd-model.cpp:1377 @b1b770d90): pocket-tts's
    /// encoder and decoder share a layout under different prefixes.
    pub fn loadSeanet(self: *TensorLoader, seanet: *mod.model.Seanet, is_decoder: bool) Error!void {
        const hparams = &self.model.hparams;
        const conv_in: [*:0]const u8 = if (is_decoder) mod.names.TN_A_GEN_WAV_SEANET_CONV_IN else mod.names.TN_A_SEANET_CONV_IN;
        const conv_out: [*:0]const u8 = if (is_decoder) mod.names.TN_A_GEN_WAV_SEANET_CONV_OUT else mod.names.TN_A_SEANET_CONV_OUT;
        const res1: [*:0]const u8 = if (is_decoder) mod.names.TN_A_GEN_WAV_SEANET_RES_CONV1 else mod.names.TN_A_SEANET_RES_CONV1;
        const res2: [*:0]const u8 = if (is_decoder) mod.names.TN_A_GEN_WAV_SEANET_RES_CONV2 else mod.names.TN_A_SEANET_RES_CONV2;
        const scale: [*:0]const u8 = if (is_decoder) mod.names.TN_A_GEN_WAV_SEANET_SCALE_CONV else mod.names.TN_A_SEANET_SCALE_CONV;

        var nb: [256]u8 = undefined;
        seanet.conv_in_w = try self.getTensor(fmtRt(&nb, conv_in, .{@as([*:0]const u8, "weight")}), true);
        seanet.conv_in_b = try self.getTensor(fmtRt(&nb, conv_in, .{@as([*:0]const u8, "bias")}), true);
        seanet.conv_out_w = try self.getTensor(fmtRt(&nb, conv_out, .{@as([*:0]const u8, "weight")}), true);
        seanet.conv_out_b = try self.getTensor(fmtRt(&nb, conv_out, .{@as([*:0]const u8, "bias")}), true);

        try seanet.stages.resize(std.heap.c_allocator, @intCast(hparams.seanet_n_stage));
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

// `mtmd_format` into a caller's buffer, for the key and tensor names: one
// helper per argument shape the C++ passes.

/// `mtmd_format(fmt, s)`.
pub fn fmtKey(buf: *[256]u8, comptime fmt: [*:0]const u8, s: [*:0]const u8) [*:0]const u8 {
    _ = stdlib.libc.snprintf(buf, buf.len, fmt, s);
    return @ptrCast(buf);
}

/// `mtmd_format(fmt, s1, s2)`.
pub fn fmtKey2(buf: *[256]u8, comptime fmt: [*:0]const u8, s1: [*:0]const u8, s2: [*:0]const u8) [*:0]const u8 {
    _ = stdlib.libc.snprintf(buf, buf.len, fmt, s1, s2);
    return @ptrCast(buf);
}

/// `mtmd_format(fmt, prefix, il, suffix)`: a layer tensor.
pub fn fmtL(buf: *[256]u8, comptime fmt: [*:0]const u8, prefix: [*:0]const u8, il: c_int, suffix: [*:0]const u8) [*:0]const u8 {
    _ = stdlib.libc.snprintf(buf, buf.len, fmt, prefix, il, suffix);
    return @ptrCast(buf);
}

/// `mtmd_format(fmt, prefix, il)`.
pub fn fmtPI(buf: *[256]u8, comptime fmt: [*:0]const u8, prefix: [*:0]const u8, il: c_int) [*:0]const u8 {
    _ = stdlib.libc.snprintf(buf, buf.len, fmt, prefix, il);
    return @ptrCast(buf);
}

/// `mtmd_format(fmt, il, suffix)`.
pub fn fmtIS(buf: *[256]u8, comptime fmt: [*:0]const u8, il: c_int, suffix: [*:0]const u8) [*:0]const u8 {
    _ = stdlib.libc.snprintf(buf, buf.len, fmt, il, suffix);
    return @ptrCast(buf);
}

/// `mtmd_format` with a format known only at run time.
pub fn fmtRt(buf: *[256]u8, fmt: [*:0]const u8, args: anytype) [*:0]const u8 {
    _ = @call(.auto, stdlib.libc.snprintf, .{ buf, buf.len, fmt } ++ args);
    return @ptrCast(buf);
}

// -----------------------------------------------------------------------------
// loading

/// Ports `mtmd_model_params` (mtmd-model.h:1307 @b1b770d90).
pub const ModelParams = struct {
    // custom value provided by user, -1 if not set
    image_min_tokens: i32 = -1,
    image_max_tokens: i32 = -1,
    no_alloc: bool = false, // only create the tensors, do not read their data
    progress_callback: ?*const fn (f32, ?*anyopaque) callconv(.c) bool = null,
    progress_callback_user_data: ?*anyopaque = null,
};

/// Ports `mtmd_models` (mtmd-model.h:1317 @b1b770d90): the modalities of an
/// mmproj file, a missing one is null.
pub const Models = struct {
    v: ?*mod.model.Model = null,
    a: ?*mod.model.Model = null,
    gen_a: ?*mod.model.Model = null,

    /// The `unique_ptr`s' destructors, in reverse member order.
    pub fn deinit(self: *Models) void {
        if (self.gen_a) |m| destroyModel(m);
        if (self.a) |m| destroyModel(m);
        if (self.v) |m| destroyModel(m);
        self.* = .{};
    }
};

/// `delete` of an `mtmd_model`.
pub fn destroyModel(m: *mod.model.Model) void {
    m.deinit();
    std.heap.c_allocator.destroy(m);
}

/// Ports `mtmd_model_load` (mtmd-model.cpp:3726 @b1b770d90): every modality
/// of `fname`, the weights allocated in `buft`. Throws on error.
pub fn modelLoad(fname: [:0]const u8, params: *const ModelParams, buft: mod.impl.c.ggml_backend_buffer_type_t) Error!Models {
    var loader = try Loader.init(fname, false, params.progress_callback, params.progress_callback_user_data);
    defer loader.deinit();

    var res: Models = .{};
    errdefer res.deinit();
    if (loader.has_vision) {
        res.v = try loadModality(&loader, params, buft, .vision);
    }

    // TODO: we don't support audio for Gemma 3N, but GGUF contains audio tensors
    // we can remove this check when we implement audio support for Gemma 3N
    const skip_audio = res.v != null and res.v.?.proj_type == .gemma3nv;
    if (loader.has_audio and !skip_audio) {
        res.a = try loadModality(&loader, params, buft, .audio);
    }

    if (loader.has_gen_audio) {
        res.gen_a = try loadModality(&loader, params, buft, .gen_audio);
    }

    return res;
}

/// Ports the `load` lambda (mtmd-model.cpp:3732 @b1b770d90) of `mtmd_model_load`.
fn loadModality(loader: *Loader, params: *const ModelParams, buft: mod.impl.c.ggml_backend_buffer_type_t, modality: mod.impl.Modality) Error!*mod.model.Model {
    const model = try std.heap.c_allocator.create(mod.model.Model);
    model.* = .{};
    errdefer destroyModel(model);
    model.hparams.custom_image_min_tokens = params.image_min_tokens;
    model.hparams.custom_image_max_tokens = params.image_max_tokens;
    try loader.loadHparams(model, modality);
    try loader.loadTensors(model, buft, params.no_alloc);
    return model;
}

/// Ports `mtmd_model_get_caps` (mtmd-model.cpp:3760 @b1b770d90): the
/// modalities of an mmproj file, without loading its tensors. A load failure
/// is the C++'s escaping exception, which `mtmd_get_cap_from_file` catches.
pub fn modelGetCaps(fname: [:0]const u8) Error!ModelCaps {
    var loader = try Loader.init(fname, true, null, null);
    defer loader.deinit();
    return .{ .has_vision = loader.has_vision, .has_audio = loader.has_audio, .has_gen_audio = loader.has_gen_audio };
}

// -----------------------------------------------------------------------------
// mtmd_model's methods

/// Ports `image_max_tokens` (mtmd-model.cpp:3773 @b1b770d90), a member of
/// `mtmd_model`: the most output tokens one image can take, -1 where the size
/// is not dynamic.
pub fn imageMaxTokens(model: *const mod.model.Model) c_int {
    const hparams = &model.hparams;
    if (model.proj_type == .deepseek4v or model.proj_type == .deepseek41v) {
        return hparams.dsv4_max_n_token;
    }
    if (hparams.image_max_pixels <= 0) {
        return -1;
    }
    const patch_area = hparams.patch_size * hparams.patch_size * hparams.n_merge * hparams.n_merge;
    return @divTrunc(hparams.image_max_pixels, patch_area);
}

/// Ports `n_output_tokens_x` (mtmd-model.cpp:3784 @b1b770d90), a member of
/// `mtmd_model`: for M-RoPE, the token positions in X; otherwise every token.
pub fn nOutputTokensX(model: *const mod.model.Model, img: *const mod.impl.ImageF32) c_int {
    const params = &model.hparams;
    const n_total = nOutputTokens(model, img);
    switch (model.proj_type) {
        .qwen2vl, .qwen25vl, .qwen3vl, .magevl, .sensenova_u1, .ling3vl, .exaone4_5, .mimovl, .glm4v, .glm5v, .paddleocr, .hunyuanvl, .youtuvl, .muse_glimmer, .lfm3vl => return @divTrunc(@divTrunc(img.nx(), params.patch_size), 2),
        .step3vl, .minicpmv4_6, .minicpmv4_7, .minicpm_ocr_1_0 => return @divTrunc(img.nx(), params.patch_size * params.n_merge),
        .deepseekocr, .deepseekocr2 => return @divTrunc(@divTrunc(img.nx(), params.patch_size), 4),
        else => {},
    }
    return n_total;
}

/// Ports `n_output_tokens_y` (mtmd-model.cpp:3819 @b1b770d90), a member of
/// `mtmd_model`.
pub fn nOutputTokensY(model: *const mod.model.Model, img: *const mod.impl.ImageF32) c_int {
    const params = &model.hparams;
    switch (model.proj_type) {
        .qwen2vl, .qwen25vl, .qwen3vl, .magevl, .sensenova_u1, .ling3vl, .exaone4_5, .mimovl, .glm4v, .glm5v, .paddleocr, .hunyuanvl, .youtuvl, .muse_glimmer, .lfm3vl => return @divTrunc(@divTrunc(img.ny(), params.patch_size), 2),
        .step3vl, .minicpmv4_6, .minicpmv4_7, .minicpm_ocr_1_0 => return @divTrunc(img.ny(), params.patch_size * params.n_merge),
        else => {},
    }
    return 1;
}

/// Ports `n_output_tokens` (mtmd-model.cpp:3850 @b1b770d90), a member of
/// `mtmd_model`: the per-projector body is `arms_inputs.nOutputTokens`.
pub fn nOutputTokens(model: *const mod.model.Model, img: *const mod.impl.ImageF32) c_int {
    return mod.arms_inputs.nOutputTokens(model, img);
}

/// Ports `cbx_read_tensor` (mtmd-model.cpp:4197 @b1b770d90), a member of
/// `mtmd_model`: reads a chatterbox tensor by name as F32, returns the
/// element count (0 if not found); `out` may be null to query the size.
pub fn cbxReadTensor(model: *const mod.model.Model, name: []const u8, out: ?[*]f32, n_max: usize) usize {
    const t = (model.cbx_tensors.get(name) orelse return 0) orelse return 0;
    const n: usize = @intCast(mod.impl.c.ggml_nelements(t));
    const o = out orelse return n;
    if (n_max < n) return 0;
    if (t.type == mod.impl.c.GGML_TYPE_F32) {
        mod.impl.c.ggml_backend_tensor_get(t, o, 0, n * @sizeOf(f32));
    } else if (t.type == mod.impl.c.GGML_TYPE_F16) {
        const tmp = std.heap.c_allocator.alloc(mod.impl.c.ggml_fp16_t, n) catch mod.impl.abortUncaught("std::bad_alloc", "std::bad_alloc");
        defer std.heap.c_allocator.free(tmp);
        mod.impl.c.ggml_backend_tensor_get(t, tmp.ptr, 0, n * @sizeOf(mod.impl.c.ggml_fp16_t));
        for (0..n) |i| o[i] = mod.impl.c.ggml_fp16_to_fp32(tmp[i]);
    } else {
        return 0;
    }
    return n;
}

/// Ports `n_mmproj_embd` (mtmd-model.cpp:4223 @b1b770d90), a member of
/// `mtmd_model`: the embedding size of the projector's output, which must
/// be the text model's; the per-projector body is `arms_tensors.nMmprojEmbd`.
pub fn nMmprojEmbd(model: *const mod.model.Model) c_int {
    return mod.arms_tensors.nMmprojEmbd(model);
}

/// Ports `n_temporal_merge` (mtmd-model.cpp:4360 @b1b770d90), a member of
/// `mtmd_model`: the qwen-vl models and Nemotron Omni fuse two frames into one.
pub fn nTemporalMerge(model: *const mod.model.Model) c_int {
    return switch (model.proj_type) {
        .qwen2vl, .qwen25vl, .qwen3vl, .lfm3vl, .nemotron_omni, .ling3vl => 2,
        else => 1,
    };
}

/// What the input arms of `encode` (mtmd-subcontext.cpp:521 @b1b770d90), a
/// member of `mtmd_subcontext`, read and the helpers they call.
pub const EncodeState = struct {
    ctx: *Ctx,
    params: *EncodeParams,
    imgs: *const mod.impl.ImageF32Batch,
    gf: ?*mod.impl.c.ggml_cgraph,

    image_size_width: c_int,
    image_size_height: c_int,
    patch_size: c_int,
    num_patches: c_int,
    n_pos: c_int,
    pos_w: c_int,
    pos_h: c_int,

    /// The `get_inp_tensor` lambda (mtmd-subcontext.cpp:578 @b1b770d90).
    pub fn getInpTensor(self: *const EncodeState, name: [*:0]const u8) mod.graph.T {
        const inp = mod.impl.c.ggml_graph_get_tensor(self.gf, name);
        if (inp == null) _ = mod.impl.c.ggml_abort("mtmd-subcontext.cpp", 581, "Failed to get tensor %s", name);
        if (inp.*.flags & mod.impl.c.GGML_TENSOR_FLAG_INPUT == 0) _ = mod.impl.c.ggml_abort("mtmd-subcontext.cpp", 584, "Tensor %s is not an input tensor", name);
        return inp;
    }

    /// The `set_input_f32` lambda (mtmd-subcontext.cpp:589 @b1b770d90).
    pub fn setInputF32(self: *const EncodeState, name: [*:0]const u8, values: []const f32) void {
        const cur = self.getInpTensor(name);
        mod.impl.ggmlAssert(cur.?.type == mod.impl.c.GGML_TYPE_F32, "mtmd-subcontext.cpp", 591, "cur->type == GGML_TYPE_F32");
        mod.impl.ggmlAssert(mod.impl.c.ggml_nelements(cur) == @as(i64, @intCast(values.len)), "mtmd-subcontext.cpp", 592, "ggml_nelements(cur) == (int64_t)values.size()");
        mod.impl.c.ggml_backend_tensor_set(cur, values.ptr, 0, mod.impl.c.ggml_nbytes(cur));
    }

    /// The `set_input_attn_mask` lambda (mtmd-subcontext.cpp:597 @b1b770d90): a
    /// mask from `build_inp_attn_mask`, f16 if flash attn is enabled.
    pub fn setInputAttnMask(self: *const EncodeState, values: []const f32, idx: c_int) Error!void {
        var buf: [32]u8 = undefined;
        const cur = self.getInpTensor(mod.graph.attnMaskName(&buf, idx));
        mod.impl.ggmlAssert(mod.impl.c.ggml_nelements(cur) == @as(i64, @intCast(values.len)), "mtmd-subcontext.cpp", 599, "ggml_nelements(cur) == (int64_t)values.size()");
        if (cur.?.type == mod.impl.c.GGML_TYPE_F16) {
            const values_f16 = try std.heap.c_allocator.alloc(mod.impl.c.ggml_fp16_t, values.len);
            defer std.heap.c_allocator.free(values_f16);
            mod.impl.c.ggml_fp32_to_fp16_row(values.ptr, values_f16.ptr, @intCast(values.len));
            mod.impl.c.ggml_backend_tensor_set(cur, values_f16.ptr, 0, mod.impl.c.ggml_nbytes(cur));
        } else {
            mod.impl.ggmlAssert(cur.?.type == mod.impl.c.GGML_TYPE_F32, "mtmd-subcontext.cpp", 605, "cur->type == GGML_TYPE_F32");
            mod.impl.c.ggml_backend_tensor_set(cur, values.ptr, 0, mod.impl.c.ggml_nbytes(cur));
        }
    }

    /// The `set_input_i32` lambda (mtmd-subcontext.cpp:610 @b1b770d90).
    pub fn setInputI32(self: *const EncodeState, name: [*:0]const u8, values: []const i32) void {
        const cur = self.getInpTensor(name);
        mod.impl.ggmlAssert(cur.?.type == mod.impl.c.GGML_TYPE_I32, "mtmd-subcontext.cpp", 612, "cur->type == GGML_TYPE_I32");
        mod.impl.ggmlAssert(mod.impl.c.ggml_nelements(cur) == @as(i64, @intCast(values.len)), "mtmd-subcontext.cpp", 613, "ggml_nelements(cur) == (int64_t)values.size()");
        mod.impl.c.ggml_backend_tensor_set(cur, values.ptr, 0, mod.impl.c.ggml_nbytes(cur));
    }

    /// The `set_gen_state_in` lambda (mtmd-subcontext.cpp:618 @b1b770d90): the
    /// state of the previous call, zero on a cold start; an in-place slot is
    /// the graph's to read.
    pub fn setGenStateIn(self: *const EncodeState) Error!void {
        const state_slots = self.params.gen_state.?.slots.items;
        const slots = try mod.models.genStateSlots(&self.ctx.model.hparams, self.ctx.model);
        defer mod.models.freeSlots(slots);
        var i: usize = 0;
        for (slots) |slot| {
            mod.impl.ggmlAssert(i < state_slots.len, "mtmd-subcontext.cpp", 622, "i < slots.size()");
            if (slot.in_place) {
                i += 1;
                continue;
            }
            var nb: [256]u8 = undefined;
            _ = stdlib.libc.snprintf(&nb, nb.len, "state_in_%s", slot.name.ptr);
            mod.impl.c.ggml_backend_tensor_copy(state_slots[i], self.getInpTensor(@ptrCast(&nb)));
            i += 1;
        }
    }

    /// The `set_enc_state_in` lambda (mtmd-subcontext.cpp:632 @b1b770d90):
    /// the streaming encoder's state in, and its "pos" slot returned; without
    /// a stream the state is zero.
    pub fn setEncStateIn(self: *const EncodeState) Error!c_int {
        const state_slots = try mod.models.encStateSlots(&self.ctx.model.hparams, self.ctx.model);
        defer mod.models.freeSlots(state_slots);
        mod.impl.ggmlAssert(state_slots.len != 0 and std.mem.eql(u8, state_slots[0].name, "pos"), "mtmd-subcontext.cpp", 634, "!state_slots.empty() && state_slots[0].name == \"pos\"");
        var nb: [256]u8 = undefined;
        if (self.params.enc_state == null) {
            for (state_slots) |slot| {
                _ = stdlib.libc.snprintf(&nb, nb.len, "state_in_%s", slot.name.ptr);
                const t = self.getInpTensor(@ptrCast(&nb));
                mod.impl.c.ggml_backend_tensor_memset(t, 0, 0, mod.impl.c.ggml_nbytes(t));
            }
            return 0;
        }
        const slots = self.params.enc_state.?.slots.items;
        mod.impl.ggmlAssert(slots.len == state_slots.len, "mtmd-subcontext.cpp", 643, "slots.size() == state_slots.size()");
        var base_f: f32 = 0.0;
        mod.impl.c.ggml_backend_tensor_get(slots[0], &base_f, 0, @sizeOf(f32));
        for (state_slots, 0..) |slot, i| {
            _ = stdlib.libc.snprintf(&nb, nb.len, "state_in_%s", slot.name.ptr);
            mod.impl.c.ggml_backend_tensor_copy(slots[i], self.getInpTensor(@ptrCast(&nb)));
        }
        return @intFromFloat(base_f);
    }

    /// The `set_lfm3a_window_inputs` lambda (mtmd-subcontext.cpp:654 @b1b770d90): the lfm3a
    /// encoder's and codec's causal sliding window over [cached window | this call].
    pub fn setLfm3aWindowInputs(self: *const EncodeState, base: c_int, n_pos: c_int, window: c_int) Error!void {
        const positions = try std.heap.c_allocator.alloc(i32, @intCast(n_pos));
        defer std.heap.c_allocator.free(positions);
        for (positions, 0..) |*v, i| v.* = base + @as(i32, @intCast(i));
        self.setInputI32("positions", positions);

        // key j holds position base - prefix + j
        const prefix = window - 1;
        const n_kv = prefix + n_pos;
        const mask = try std.heap.c_allocator.alloc(f32, @as(usize, @intCast(n_kv)) * @as(usize, @intCast(n_pos)));
        defer std.heap.c_allocator.free(mask);
        @memset(mask, -std.math.inf(f32));
        var q: c_int = 0;
        while (q < n_pos) : (q += 1) {
            var j: c_int = @max(0, q + prefix - window + 1);
            while (j <= q + prefix) : (j += 1) {
                if (base - prefix + j >= 0) mask[@as(usize, @intCast(q)) * @as(usize, @intCast(n_kv)) + @as(usize, @intCast(j))] = 0.0;
            }
        }
        try self.setInputAttnMask(mask, 0);
    }

    /// The `set_pockettts_tfm_inputs` lambda (mtmd-subcontext.cpp:677 @b1b770d90): rope
    /// positions and the causal sliding-window mask of the mimi transformers.
    pub fn setPocketttsTfmInputs(self: *const EncodeState) Error!void {
        const hparams = &self.ctx.model.hparams;
        const n_pos = mod.impl.c.ggml_nelements(self.getInpTensor("inp_pos"));
        mod.impl.ggmlAssert(n_pos > 0, "mtmd-subcontext.cpp", 679, "n_pos > 0");
        const positions = try std.heap.c_allocator.alloc(i32, @intCast(n_pos));
        defer std.heap.c_allocator.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(i);
        self.setInputI32("inp_pos", positions);

        // the preprocessor truncates the waveform to keep this mask bounded
        const max_pos: i64 = @divTrunc(@as(i64, mod.model.Hparams.pockettts_max_spk_seconds) * hparams.audio_sample_rate, 120);
        mod.impl.ggmlAssert(n_pos <= max_pos, "mtmd-subcontext.cpp", 688, "n_pos <= max_pos && \"pocket-tts speaker reference too long for a dense mask\"");

        const context: i64 = hparams.mimi_tfm_context;
        const mask = try std.heap.c_allocator.alloc(f32, @intCast(n_pos * n_pos));
        defer std.heap.c_allocator.free(mask);
        @memset(mask, -std.math.inf(f32));
        var q: i64 = 0;
        while (q < n_pos) : (q += 1) {
            var k: i64 = 0;
            while (k < n_pos) : (k += 1) {
                const delta = q - k;
                if (delta >= 0 and delta < context) mask[@intCast(q * n_pos + k)] = 0.0;
            }
        }
        try self.setInputAttnMask(mask, 0);
    }
};

/// Ports `copy_gen_state_out` (mtmd-subcontext.cpp:409 @b1b770d90): state_out can have another
/// shape than its slot, only the byte size must match.
fn copyGenStateOut(src: *mod.impl.c.ggml_tensor, dst: *mod.impl.c.ggml_tensor) Error!void {
    mod.impl.ggmlAssert(mod.impl.c.ggml_nbytes(src) == mod.impl.c.ggml_nbytes(dst), "mtmd-subcontext.cpp", 410, "ggml_nbytes(src) == ggml_nbytes(dst)");
    if (src.type == dst.type and mod.impl.c.ggml_are_same_shape(src, dst) and mod.impl.c.ggml_is_contiguous(src)) {
        mod.impl.c.ggml_backend_tensor_copy(src, dst);
        return;
    }
    const tmp = try std.heap.c_allocator.alloc(u8, mod.impl.c.ggml_nbytes(src));
    defer std.heap.c_allocator.free(tmp);
    mod.impl.c.ggml_backend_tensor_get(src, tmp.ptr, 0, tmp.len);
    mod.impl.c.ggml_backend_tensor_set(dst, tmp.ptr, 0, tmp.len);
}

/// Allocates one F32 tensor per slot on the shared backend, named by `fmt`
/// over the slot, and clears them: the tail `gen_state_init` and
/// `enc_state_init`'s lfm3a and gemma4a arm share (a zero state is the cold
/// start).
fn allocStateSlots(ctx: *Ctx, slots: []const mod.models.StateSlot, comptime fmt: [*:0]const u8, ctx_data: *?*mod.impl.c.ggml_context, buf: *mod.impl.c.ggml_backend_buffer_t, out: *std.ArrayList(*mod.impl.c.ggml_tensor)) Error!bool {
    const params: mod.impl.c.ggml_init_params = .{
        .mem_size = mod.impl.c.ggml_tensor_overhead() * slots.len,
        .mem_buffer = null,
        .no_alloc = true,
    };
    ctx_data.* = mod.impl.c.ggml_init(params);
    for (slots) |slot| {
        const t = mod.impl.c.ggml_new_tensor_2d(ctx_data.*, mod.impl.c.GGML_TYPE_F32, slot.ne0, slot.ne1);
        _ = mod.impl.c.ggml_format_name(t, fmt, slot.name.ptr);
        try out.append(std.heap.c_allocator, t.?);
    }
    buf.* = mod.impl.c.ggml_backend_alloc_ctx_tensors(ctx_data.*, ctx.backends.backend);
    if (buf.* == null) return false;
    // a zero state is the cold start
    mod.impl.c.ggml_backend_buffer_clear(buf.*, 0);
    return true;
}

/// Ports `gen_state_init` (mtmd-subcontext.cpp:420 @b1b770d90), a member of
/// `mtmd_subcontext`: the RNG and the slots kept between GEN_WAV calls. A
/// seed of `UINT32_MAX` means random; free the result with `destroy`.
pub fn genStateInit(ctx: *Ctx, seed: u32) Error!?*GenState {
    const st = try std.heap.c_allocator.create(GenState);
    st.* = .{};
    errdefer st.destroy();
    st.rng = stdlib.libcxx.Mt19937.init(if (seed == std.math.maxInt(u32)) stdlib.libcxx.randomDevice() else seed);

    const slots = try mod.models.genStateSlots(&ctx.model.hparams, ctx.model);
    defer mod.models.freeSlots(slots);
    if (slots.len == 0) return st;

    if (!try allocStateSlots(ctx, slots, "gen_state_%s", &st.ctx_data, &st.buf, &st.slots)) {
        mod.impl.logErr("%s: failed to allocate the gen state\n", .{"gen_state_init"});
        st.destroy();
        return null;
    }
    return st;
}

/// Ports `enc_state_init` (mtmd-subcontext.cpp:451 @b1b770d90), a member of
/// `mtmd_subcontext`: the persistent state of a causal audio encoder, one
/// per stream. `n_frames_max` is the maximum number of encoder frames per
/// call; free the result with `destroy`.
pub fn encStateInit(ctx: *Ctx, n_frames_max: i32) Error!?*EncState {
    const func = "enc_state_init";
    const hparams = &ctx.model.hparams;
    if (ctx.projType() == .lfm3a or ctx.projType() == .gemma4a) {
        // see list_enc_state_slots()
        const slots = try mod.models.encStateSlots(hparams, ctx.model);
        defer mod.models.freeSlots(slots);
        const st = try std.heap.c_allocator.create(EncState);
        st.* = .{};
        errdefer st.destroy();
        if (!try allocStateSlots(ctx, slots, "enc_state_%s", &st.ctx_data, &st.buf, &st.slots)) {
            mod.impl.logErr("%s: failed to allocate the encoder state\n", .{func});
            st.destroy();
            return null;
        }
        return st;
    }
    if (ctx.projType() != .voxtral_rt or hparams.attn_window_size <= 0) {
        mod.impl.logErr("%s: the encoder does not support streaming input\n", .{func});
        return null;
    }
    if (n_frames_max <= 0) {
        mod.impl.logErr("%s: invalid n_frames_max %d\n", .{ func, @as(c_int, n_frames_max) });
        return null;
    }

    const n_layer = hparams.n_layer;
    const d_head: i32 = if (hparams.n_embd_head > 0) hparams.n_embd_head else @divTrunc(hparams.n_embd, hparams.n_head);
    const n_embd_kv = hparams.n_head_kv * d_head;

    const st = try std.heap.c_allocator.create(EncState);
    st.* = .{};
    errdefer st.destroy();
    st.n_ctx = hparams.attn_window_size + n_frames_max;
    try st.slot_pos.appendNTimes(std.heap.c_allocator, -1, @intCast(st.n_ctx));

    const params: mod.impl.c.ggml_init_params = .{
        .mem_size = mod.impl.c.ggml_tensor_overhead() * 2 * @as(usize, @intCast(n_layer)),
        .mem_buffer = null,
        .no_alloc = true,
    };
    st.ctx_data = mod.impl.c.ggml_init(params);
    var il: c_int = 0;
    while (il < n_layer) : (il += 1) {
        const k = mod.impl.c.ggml_new_tensor_2d(st.ctx_data, mod.impl.c.GGML_TYPE_F16, n_embd_kv, st.n_ctx);
        const v = mod.impl.c.ggml_new_tensor_2d(st.ctx_data, mod.impl.c.GGML_TYPE_F16, n_embd_kv, st.n_ctx);
        _ = mod.impl.c.ggml_format_name(k, "enc_state_k_%d", il);
        _ = mod.impl.c.ggml_format_name(v, "enc_state_v_%d", il);
        try st.k.append(std.heap.c_allocator, k.?);
        try st.v.append(std.heap.c_allocator, v.?);
    }
    st.buf = mod.impl.c.ggml_backend_alloc_ctx_tensors(st.ctx_data, ctx.backends.backend);
    if (st.buf == null) {
        mod.impl.logErr("%s: failed to allocate the encoder state\n", .{func});
        st.destroy();
        return null;
    }
    // empty slots are masked out, but they must not hold NaN
    mod.impl.c.ggml_backend_buffer_clear(st.buf, 0);

    mod.impl.logInf("%s: encoder state: %d slots, %.2f MiB\n", .{ func, @as(c_int, st.n_ctx), @as(f64, @floatFromInt(mod.impl.c.ggml_backend_buffer_get_size(st.buf))) / 1024.0 / 1024.0 });

    return st;
}

/// Ports `encode` (mtmd-subcontext.cpp:521 @b1b770d90), a member of
/// `mtmd_subcontext`; the per-projector input switch is
/// `arms_inputs.setInputs`.
pub fn encode(ctx: *Ctx, params: *EncodeParams) Error!bool {
    const func = "encode";
    const imgs = params.imgs.?;
    const n_batch_cur: c_int = @intCast(imgs.entries.items.len);

    // [QWEN_VIDEO] for video models, the batch dimension is used as temporal dimension for merged frames
    if (!ctx.support_batch and n_batch_cur > nTemporalMerge(ctx.model)) {
        mod.impl.logErr("%s: batch size %d exceeds maximum supported batch/temporal-merge size %d\n", .{ func, n_batch_cur, nTemporalMerge(ctx.model) });
        return false;
    }

    // if buffers are not allocated, we need to do a warmup run to allocate them
    if (!ctx.is_allocated) try ctx.warmup(imgs);

    const needs_gen_state = params.gen_process == .gen_code or
        params.gen_process == .gen_wav or
        params.gen_process == .tts;
    if (needs_gen_state and params.gen_state == null) {
        mod.impl.logErr("%s: audio gen requires a gen state\n", .{func});
        return false;
    }
    if (ctx.projType() == .drama3_gen and params.gen_process == .gen_code) {
        var pos: f32 = 0.0;
        mod.impl.c.ggml_backend_tensor_get(params.gen_state.?.slots.items[0], &pos, 0, @sizeOf(f32));
        if (@as(c_int, @intFromFloat(pos)) >= mod.models.drama3_gen.n_pos_max) {
            mod.impl.logErr("%s: the codebook tower holds at most %d rows\n", .{ func, @as(c_int, mod.models.drama3_gen.n_pos_max) });
            return false;
        }
    }

    // build the inference graph
    mod.impl.c.ggml_backend_sched_reset(ctx.backends.sched);
    var built = try mod.models.buildGraph(ctx, imgs, params);
    defer built.deinit();
    const gf = built.gf;

    // the last node is the embedding tensor, code2wav has no out_embd
    // read it before scheduling: backends may reorder the nodes of graphs with several outputs (Metal)
    const embeddings: mod.graph.T = if (params.out_embd != null) mod.impl.c.ggml_graph_node(gf, -1) else null;

    if (!mod.impl.c.ggml_backend_sched_alloc_graph(ctx.backends.sched, gf)) {
        mod.impl.logErr("%s: failed to allocate compute graph\n", .{func});
        return false;
    }

    // set inputs
    const model = ctx.model;
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
        const inp_raw = try std.heap.c_allocator.alloc(f32, nelem);
        defer std.heap.c_allocator.free(inp_raw);
        @memset(inp_raw, 0);

        // planar R, G, B per image; the batch dim is the temporal dim in Qwen-VL models
        {
            const nx: usize = @intCast(imgs.entries.items[0].nx());
            const ny: usize = @intCast(imgs.entries.items[0].ny());
            const n = nx * ny;

            var b: c_int = 0;
            while (b < n_batch_cur) : (b += 1) {
                mod.impl.logDbg("%s: copying image %d/%d to input buffer (nx=%d, ny=%d)\n", .{ func, b + 1, n_batch_cur, @as(c_int, @intCast(nx)), @as(c_int, @intCast(ny)) });
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
    } else if (params.gen_process != .gen_wav and
        params.gen_process != .tts and
        params.gen_process != .tts_vocode and
        mod.impl.c.ggml_graph_get_tensor(gf, "inp_raw") != null)
    {
        // audio input. GEN_WAV and the chatterbox TTS stages are not here: they take codes or feats, set in the switch below.
        // a GEN_CODE graph may have no hidden-state input (lfm3a_gen feedback embedding)
        mod.impl.ggmlAssert(imgs.entries.items.len == 1, "mtmd-subcontext.cpp", 752, "imgs.entries.size() == 1");

        const mel_inp = &imgs.entries.items[0];
        const buf = mel_inp.roBuf();
        const n_step = mel_inp.nx();
        const n_mel = mel_inp.ny();
        mod.impl.ggmlAssert(@as(usize, @intCast(n_step)) * @as(usize, @intCast(n_mel)) == buf.len, "mtmd-subcontext.cpp", 758, "(size_t)n_step * n_mel == buf.size()");

        st.setInputF32("inp_raw", buf);
    }

    // set input per projector; an arm that returns false is the C++'s
    // `return false` out of encode()
    if (!try mod.arms_inputs.setInputsChecked(&st)) return false;

    backendCpuSetNThreads(ctx.backends.backend_cpu, params.n_threads);

    const status = mod.impl.c.ggml_backend_sched_graph_compute(ctx.backends.sched, gf);
    if (status != mod.impl.c.GGML_STATUS_SUCCESS) {
        mod.impl.logErr("%s: ggml_backend_sched_graph_compute failed with error %d\n", .{ func, @as(c_int, @intCast(status)) });
        return false;
    }

    if (embeddings != null) {
        // sanity check (assuming that all images in batch have the same number of tokens, so we only check the first one)
        const n_tokens_out: c_int = @intCast(embeddings.?.ne[1]);
        const expected_n_tokens_out = nOutputTokens(model, &imgs.entries.items[0]);
        if (n_tokens_out != expected_n_tokens_out) {
            mod.impl.logErr("%s: expected output %d tokens, got %d\n", .{ func, expected_n_tokens_out, n_tokens_out });
            _ = mod.impl.c.ggml_abort("mtmd-subcontext.cpp", 2288, "Invalid number of output tokens");
        }

        mod.impl.logDbg("%s: output embedding shape [%d, %d, %d]\n", .{ func, @as(c_int, @intCast(embeddings.?.ne[0])), @as(c_int, @intCast(embeddings.?.ne[1])), @as(c_int, @intCast(embeddings.?.ne[2])) });

        // copy output to user buffer if provided
        // if output is empty, skip the copy
        const out_batch_embd = params.out_embd.?;
        if (out_batch_embd.items.len != 0) {
            if (out_batch_embd.items.len != @as(usize, @intCast(mod.impl.c.ggml_nelements(embeddings)))) {
                mod.impl.logErr("%s: output buffer has %zu elements but expected %zu\n", .{ func, out_batch_embd.items.len, @as(usize, @intCast(mod.impl.c.ggml_nelements(embeddings))) });
                _ = mod.impl.c.ggml_abort("mtmd-subcontext.cpp", 2300, "Output buffer size mismatch");
            }
            mod.impl.c.ggml_backend_tensor_get(embeddings, out_batch_embd.items.ptr, 0, mod.impl.c.ggml_nbytes(embeddings));
        } else {
            mod.impl.logWrn("%s: output buffer is empty, skipping copy\n", .{func});
        }
    }

    //
    // for audio gen models
    //

    // chatterbox's stages, each returns here: `MTMD_SUBCTX_GEN_PROCESS_TOKENIZE`,
    // `MTMD_SUBCTX_GEN_PROCESS_TTS_VOCODE`, `MTMD_SUBCTX_GEN_PROCESS_TTS`
    // (mtmd-subcontext.cpp:2312, 2374, 2382 @b1b770d90)
    if (ctx.projType() == .chatterbox and params.gen_process == .tokenize) return mod.models.chatterbox_gen.encodeTokenizeOut(ctx, params, gf);
    if (ctx.projType() == .chatterbox and params.gen_process == .tts_vocode) return mod.models.chatterbox_gen.encodeVocodeOut(params, gf);
    if (ctx.projType() == .chatterbox and params.gen_process == .tts) return mod.models.chatterbox_gen.encodeTtsOut(ctx, params, gf);

    // optional outputs: a pipeline yields codes or feats, and not all have an eos head
    if (params.out_codes) |out_codes| {
        const codes = mod.impl.c.ggml_graph_get_tensor(gf, "out_codes");
        if (codes != null) {
            try out_codes.resize(std.heap.c_allocator, @intCast(mod.impl.c.ggml_nelements(codes)));
            mod.impl.c.ggml_backend_tensor_get(codes, out_codes.items.ptr, 0, mod.impl.c.ggml_nbytes(codes));
        }
    }
    if (params.out_feats) |out_feats| {
        const feats = mod.impl.c.ggml_graph_get_tensor(gf, "out_feats");
        if (feats != null) {
            try out_feats.resize(std.heap.c_allocator, @intCast(mod.impl.c.ggml_nelements(feats)));
            mod.impl.c.ggml_backend_tensor_get(feats, out_feats.items.ptr, 0, mod.impl.c.ggml_nbytes(feats));
        }
    }
    if (params.out_is_eos) |out_is_eos| {
        const eos = mod.impl.c.ggml_graph_get_tensor(gf, "out_eos_score");
        if (eos != null) {
            mod.impl.ggmlAssert(mod.impl.c.ggml_nelements(eos) == 1, "mtmd-subcontext.cpp", 2536, "ggml_nelements(eos) == 1");
            var score: f32 = 0.0;
            mod.impl.c.ggml_backend_tensor_get(eos, &score, 0, @sizeOf(f32));
            out_is_eos.* = score > hparams.gen_eos_threshold;
        }
    }
    if (params.out_audio) |out_audio| {
        const audio = mod.impl.c.ggml_graph_get_tensor(gf, "out_audio");
        if (audio == null) _ = mod.impl.c.ggml_abort("mtmd-subcontext.cpp", 2545, "out_audio requested but graph has no \"out_audio\" tensor");
        try out_audio.resize(std.heap.c_allocator, @intCast(mod.impl.c.ggml_nelements(audio)));
        mod.impl.c.ggml_backend_tensor_get(audio, out_audio.items.ptr, 0, mod.impl.c.ggml_nbytes(audio));

        if (model.proj_type == .soprano) {
            const spectrum = try std.heap.c_allocator.dupe(f32, out_audio.items);
            defer std.heap.c_allocator.free(spectrum);
            try mod.models.soprano.decodeSpectrum(model, spectrum, out_audio);
        }

        // drop the tail audio that comes from the code-0 rear padding, drama3 has no padding
        if (model.proj_type != .drama3_gen) {
            const n_codes: i64 = if (params.codes != null) mod.graph.ne(model.gen_code_head_w, 2) + 1 else 0;
            const n_frames_w: i64 = hparams.wav_tfm_swa;
            const n_frames: i64 = if (params.codes) |cd| @divTrunc(@as(i64, @intCast(cd.items.len)), n_codes) else n_frames_w;
            if (n_frames < n_frames_w) {
                const hop = out_audio.items.len / @as(usize, @intCast(n_frames_w));
                out_audio.shrinkRetainingCapacity(@as(usize, @intCast(n_frames)) * hop);
            }
        }
    }
    if (params.enc_state != null and (model.proj_type == .lfm3a or model.proj_type == .gemma4a)) {
        const state_slots = params.enc_state.?.slots.items;
        const slots = try mod.models.encStateSlots(hparams, model);
        defer mod.models.freeSlots(slots);
        for (slots, 0..) |slot, i| {
            var nb: [256]u8 = undefined;
            _ = stdlib.libc.snprintf(&nb, nb.len, "state_out_%s", slot.name.ptr);
            const t = mod.impl.c.ggml_graph_get_tensor(gf, @ptrCast(&nb));
            mod.impl.ggmlAssert(t != null, "mtmd-subcontext.cpp", 2572, "t != nullptr");
            try copyGenStateOut(t, state_slots[i]);
        }
    } else if (params.enc_state != null and model.proj_type == .nemotron_transcribe) {
        const es = params.enc_state.?;
        const chunk = mod.impl.c.ggml_graph_get_tensor(gf, "spk_chunk");
        const preds = mod.impl.c.ggml_graph_get_tensor(gf, "spk_preds");
        mod.impl.ggmlAssert(chunk != null and preds != null, "mtmd-subcontext.cpp", 2579, "chunk != nullptr && preds != nullptr");
        try es.spk_chunk.resize(std.heap.c_allocator, @intCast(mod.impl.c.ggml_nelements(chunk)));
        try es.spk_preds.resize(std.heap.c_allocator, @intCast(mod.impl.c.ggml_nelements(preds)));
        mod.impl.c.ggml_backend_tensor_get(chunk, es.spk_chunk.items.ptr, 0, mod.impl.c.ggml_nbytes(chunk));
        mod.impl.c.ggml_backend_tensor_get(preds, es.spk_preds.items.ptr, 0, mod.impl.c.ggml_nbytes(preds));
    } else if (params.enc_state) |es| {
        const tail = mod.impl.c.ggml_graph_get_tensor(gf, "out_stem_tail");
        mod.impl.ggmlAssert(tail != null, "mtmd-subcontext.cpp", 2587, "tail != nullptr");
        try es.stem_tail.resize(std.heap.c_allocator, @intCast(mod.impl.c.ggml_nelements(tail)));
        mod.impl.c.ggml_backend_tensor_get(tail, es.stem_tail.items.ptr, 0, mod.impl.c.ggml_nbytes(tail));

        const n_new = @divTrunc(imgs.entries.items[0].nx() - 2, 2);
        var i: i32 = 0;
        while (i < n_new) : (i += 1) es.slot_pos.items[@intCast(@rem(es.n_past + i, es.n_ctx))] = es.n_past + i;
        es.n_past += n_new;
    }

    if (params.gen_process == .gen_wav) {
        const state_slots = params.gen_state.?.slots.items;
        const slots = try mod.models.genStateSlots(hparams, model);
        defer mod.models.freeSlots(slots);
        var i: usize = 0;
        for (slots) |slot| {
            if (slot.in_place) {
                i += 1;
                continue;
            }
            var nb: [256]u8 = undefined;
            _ = stdlib.libc.snprintf(&nb, nb.len, "state_out_%s", slot.name.ptr);
            const t = mod.impl.c.ggml_graph_get_tensor(gf, @ptrCast(&nb));
            if (t == null) _ = mod.impl.c.ggml_abort("mtmd-subcontext.cpp", 2608, "graph has no \"state_out_%s\" tensor", slot.name.ptr);
            mod.impl.ggmlAssert(i < state_slots.len, "mtmd-subcontext.cpp", 2610, "i < slots.size()");
            try copyGenStateOut(t, state_slots[i]);
            i += 1;
        }
    }

    //
    // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set
    //

    if (ctx.debug_output_embeddings and embeddings != null) {
        const n_embd = embeddings.?.ne[0];
        const n_tokens = embeddings.?.ne[1];
        const emb_data = try std.heap.c_allocator.alloc(f32, @intCast(mod.impl.c.ggml_nelements(embeddings)));
        defer std.heap.c_allocator.free(emb_data);
        mod.impl.c.ggml_backend_tensor_get(embeddings, emb_data.ptr, 0, mod.impl.c.ggml_nbytes(embeddings));

        mod.impl.logInf("\n=== MTMD_DEBUG_EMBEDDINGS ===\n", .{});
        mod.impl.logInf("Shape: [%lld, %lld]\n", .{ @as(c_longlong, n_embd), @as(c_longlong, n_tokens) });

        // TEMP debugging (parity validation), will be removed before merge
        // when the env var holds a path, dump the raw data: [int32 n_tokens][int32 n_embd][f32 data]
        if (stdlib.env.getenv("MTMD_DEBUG_EMBEDDINGS")) |dump_path| {
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
        mod.impl.logInf("Token 0 (first 16 values): ", .{});
        var i: i64 = 0;
        while (i < @min(@as(i64, 16), n_embd)) : (i += 1) mod.impl.logInf("%.6f ", .{@as(f64, emb_data[@intCast(i)])});
        mod.impl.logInf("\n", .{});

        // Print last few values of first token
        if (n_embd > 16) {
            mod.impl.logInf("Token 0 (last 16 values):  ", .{});
            i = n_embd - 16;
            while (i < n_embd) : (i += 1) mod.impl.logInf("%.6f ", .{@as(f64, emb_data[@intCast(i)])});
            mod.impl.logInf("\n", .{});
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
        mod.impl.logInf("Stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, sum=%.6f\n", .{ @as(f64, mean), @as(f64, @sqrt(@as(f32, variance))), @as(f64, min_val), @as(f64, max_val), @as(f64, sum) });
        mod.impl.logInf("=== END MTMD_DEBUG_EMBEDDINGS ===\n\n", .{});
    }

    return true;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    mod.impl.refAllDeep(@This());
}
