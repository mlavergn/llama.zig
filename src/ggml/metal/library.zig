//! The Metal pipeline cache and the 68 kernel-selection functions over it.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp` at
//! v0.3.0 (`c1d0e7a00`). Each declaration below names the C++ it replaces
//! and the line it began at.
//!
//! # What this file is
//!
//! `ggml-metal-device.cpp` shares a header with `ggml-metal-device.m` and
//! the split between them is clean — measured, no symbol defined twice.
//! The Objective-C owns the Metal objects and `ggml_metal_device_supports_op`;
//! this file owns the *names*. Every function here builds the name of an
//! MSL kernel from an op's types and shape, looks it up in the library's
//! cache, and compiles it on a miss.
//!
//! So the whole file is string formatting plus a cache, and its failure
//! mode is a mistyped kernel name. That is loud when the name does not
//! exist — `compile_pipeline` returns null and the caller aborts — and
//! quiet when it names a *different* real kernel, which is what
//! `backend-ops` and `node-diff --gpu` are for.
//!
//! # Two shapes, repeated
//!
//! Most functions are `base` = the MSL function name, `name` = the cache
//! key, then `getOrCompile`. Where a kernel takes
//! `MTLFunctionConstantValues` the key carries the constants too, because
//! two specialisations of one function are two pipelines.
//!
//! `base` and `name` are `char[256]` in the C and fixed buffers here. The
//! C truncates silently on overflow; `Name` aborts instead, since a
//! truncated kernel name would be a wrong lookup rather than a short one.

const std = @import("std");
const impl = @import("../impl.zig");
const mc = @import("device_c.zig");
const fc = @import("impl_c.zig");
const tuning = @import("tuning.zig");

const c = impl.c;
const Tensor = c.ggml_tensor;
const Pwp = mc.PipelineWithParams;

/// The C `new`s the cache and `delete`s it, and nothing hands it an
/// allocator. libc's is what a C entry point can reach — the same
/// reasoning as the rest of the port.
const allocator = std.heap.c_allocator;

/// The C's `char base[256]` / `char name[256]`.
///
/// `snprintf` truncates; this aborts. A truncated kernel name is not a
/// shorter lookup, it is a *different* one, and the C's own 256 is chosen
/// to be far larger than any name it builds — so hitting the limit means
/// a bug upstream of here, not a long name.
const Name = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,

    fn set(self: *Name, comptime fmt: []const u8, args: anytype) void {
        const out = std.fmt.bufPrint(&self.buf, fmt, args) catch
            impl.abort("metal: kernel name does not fit in 256 bytes");
        self.len = out.len;
        self.buf[self.len] = 0;
    }

    fn z(self: *const Name) [*:0]const u8 {
        return @ptrCast(&self.buf);
    }

    fn str(self: *const Name) []const u8 {
        return self.buf[0..self.len];
    }
};

/// The tail every one of the 68 functions ends with: look the name up,
/// compile it on a miss.
///
/// Parameters:
/// - `lib`: the Metal library holding the cache.
/// - `base`: the MSL function name.
/// - `name`: the cache key, which differs from `base` when the kernel is
///   specialised by constant values.
/// - `cv`: the constant values, or null.
///
/// Return: the pipeline and the parameters the library recorded with it.
fn getOrCompile(lib: *mc.Library, base: *const Name, name: *const Name, cv: ?*mc.Cv) Pwp {
    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }
    return res;
}

/// `ggml_type_name`, as a Zig slice.
fn typeName(t: c.enum_ggml_type) []const u8 {
    return std.mem.span(c.ggml_type_name(t));
}

/// `op->src[i]`, **narrowed**.
///
/// `op.src[i]` is a `[*c]ggml_tensor`, and Zig 0.16 types `p.*.ne[0]` on a
/// C pointer as the whole `[4]i64` — the miscompile `CLAUDE.md` records.
/// Every one of the 68 functions below reads `ne`, `nb` or `type` off a
/// source, so narrowing happens here once rather than being remembered 68
/// times. `.?` does **not** narrow a `[*c]`; `impl.one` does.
inline fn src(op: *const Tensor, i: usize) *const Tensor {
    return impl.one(Tensor, op.src[i]);
}

/// `op->src[i]` when it may be absent.
inline fn srcOpt(op: *const Tensor, i: usize) ?*const Tensor {
    return if (op.src[i] == null) null else impl.one(Tensor, op.src[i]);
}

// -----------------------------------------------------------------------------
// The device singleton

/// Ports `ggml_metal_device_get` (ggml-metal-device.cpp:21 @c1d0e7a00).
///
/// **The C leaks deliberately and so does this.** Its `static
/// std::vector<ggml_metal_device_ptr> devs` is appended to on *every*
/// call and the last entry returned, so repeated calls build up devices
/// that are freed only at exit. Reproduced rather than tidied: the
/// unique_ptr vector is what gives the returned pointer a lifetime longer
/// than the caller, and `ggml-metal.cpp` relies on that.
pub export fn ggml_metal_device_get(device: c_int, n_devices: c_int) callconv(.c) ?*mc.Device {
    const S = struct {
        var devs: [16]?*mc.Device = @splat(null);
        var n: usize = 0;
        var mutex: std.c.pthread_mutex_t = .{};
    };
    _ = std.c.pthread_mutex_lock(&S.mutex);
    defer _ = std.c.pthread_mutex_unlock(&S.mutex);

    const dev = mc.ggml_metal_device_init(device, n_devices);
    if (S.n < S.devs.len) {
        S.devs[S.n] = dev;
        S.n += 1;
    }
    return dev;
}

// -----------------------------------------------------------------------------
// The pipeline cache

/// Ports `struct ggml_metal_pipelines` (ggml-metal-device.cpp:29
/// @c1d0e7a00).
///
/// The C's `std::unordered_map<std::string, ggml_metal_pipeline_t>`. The
/// keys are kernel names the caller owns only transiently, so each is
/// copied in; `free` releases both the keys and the pipelines.
const Pipelines = struct {
    data: std.StringHashMapUnmanaged(*mc.Pipeline) = .empty,
};

/// Ports `ggml_metal_pipelines_init` (ggml-metal-device.cpp:33
/// @c1d0e7a00).
pub export fn ggml_metal_pipelines_init() callconv(.c) ?*mc.Pipelines {
    const res = allocator.create(Pipelines) catch impl.abort("metal: out of memory allocating the pipeline cache");
    res.* = .{};
    return @ptrCast(res);
}

/// Ports `ggml_metal_pipelines_free` (ggml-metal-device.cpp:39
/// @c1d0e7a00).
pub export fn ggml_metal_pipelines_free(ppls: ?*mc.Pipelines) callconv(.c) void {
    const p = ppls orelse return;
    const self: *Pipelines = @ptrCast(@alignCast(p));
    var it = self.data.iterator();
    while (it.next()) |entry| {
        mc.ggml_metal_pipeline_free(entry.value_ptr.*);
        allocator.free(entry.key_ptr.*);
    }
    self.data.deinit(allocator);
    allocator.destroy(self);
}

/// Ports `ggml_metal_pipelines_add` (ggml-metal-device.cpp:51
/// @c1d0e7a00).
///
/// The C's `data[name] = pipeline` replaces silently; so does this, and
/// the replaced pipeline is **not** freed — matching the C, where
/// `operator[]` overwrites the handle and leaks the old one. The only
/// caller inserts each name once.
pub export fn ggml_metal_pipelines_add(ppls: *mc.Pipelines, name: [*:0]const u8, pipeline: *mc.Pipeline) callconv(.c) void {
    const self: *Pipelines = @ptrCast(@alignCast(ppls));
    const key = std.mem.span(name);
    if (self.data.getPtr(key)) |slot| {
        slot.* = pipeline;
        return;
    }
    const owned = allocator.dupe(u8, key) catch impl.abort("metal: out of memory adding a pipeline");
    self.data.put(allocator, owned, pipeline) catch impl.abort("metal: out of memory adding a pipeline");
}

/// Ports `ggml_metal_pipelines_get` (ggml-metal-device.cpp:55
/// @c1d0e7a00).
///
/// Return: the cached pipeline, or null when the name is absent.
/// Borrowed; the cache owns it.
pub export fn ggml_metal_pipelines_get(ppls: *mc.Pipelines, name: [*:0]const u8) callconv(.c) ?*mc.Pipeline {
    const self: *Pipelines = @ptrCast(@alignCast(ppls));
    return self.data.get(std.mem.span(name));
}

// -----------------------------------------------------------------------------
// Kernel selection
//
// One function per op family, in the C's order. Each builds `base`, the
// MSL function name, and `name`, the cache key -- the same string unless
// the kernel is specialised by function constants, in which case the key
// carries them.

/// Ports `ggml_metal_library_get_pipeline_base` (ggml-metal-device.cpp:63
/// @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_base(lib: *mc.Library, op: c.enum_ggml_op) callconv(.c) Pwp {
    const op_str = switch (op) {
        c.GGML_OP_ADD_ID => "add_id",
        else => impl.abort("metal: no base pipeline for this op"),
    };

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_{s}", .{op_str});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_cpy` (ggml-metal-device.cpp:84
/// @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_cpy(lib: *mc.Library, tsrc: c.enum_ggml_type, tdst: c.enum_ggml_type) callconv(.c) Pwp {
    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_cpy_{s}_{s}", .{ typeName(tsrc), typeName(tdst) });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// The body `ggml_metal_library_get_pipeline_pool_1d` and
/// `ggml_metal_library_get_pipeline_pool_2d` share
/// (ggml-metal-device.cpp:99, 124 @c1d0e7a00) -- the C writes it out
/// twice, differing only in the `1d` or `2d` in the kernel name.
fn poolPipeline(comptime dim: []const u8, lib: *mc.Library, op: *const Tensor, op_pool: c.enum_ggml_op_pool) Pwp {
    const src0 = src(op, 0);
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(op->src[0])");
    impl.assert(src0.type == c.GGML_TYPE_F32 and src0.type == op.type, "src0 and dst are f32");

    const pool_str = switch (op_pool) {
        c.GGML_OP_POOL_AVG => "avg",
        c.GGML_OP_POOL_MAX => "max",
        else => impl.abort("metal: pool op not implemented"),
    };

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_pool_{s}_{s}_{s}", .{ dim, pool_str, typeName(src0.type) });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_pool_1d` (ggml-metal-device.cpp:99
/// @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_pool_1d(lib: *mc.Library, op: *const Tensor, op_pool: c.enum_ggml_op_pool) callconv(.c) Pwp {
    return poolPipeline("1d", lib, op, op_pool);
}

/// Ports `ggml_metal_library_get_pipeline_pool_2d` (ggml-metal-device.cpp:124
/// @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_pool_2d(lib: *mc.Library, op: *const Tensor, op_pool: c.enum_ggml_op_pool) callconv(.c) Pwp {
    return poolPipeline("2d", lib, op, op_pool);
}

/// Ports `ggml_metal_library_get_pipeline_get_rows`
/// (ggml-metal-device.cpp:149 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_get_rows(lib: *mc.Library, tsrc: c.enum_ggml_type) callconv(.c) Pwp {
    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_get_rows_{s}", .{typeName(tsrc)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_set_rows`
/// (ggml-metal-device.cpp:164 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_set_rows(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const tsrc = src(op, 0).type;
    const tidx = src(op, 1).type;
    const tdst = op.type;

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_set_rows_{s}_{s}_{s}", .{ typeName(tsrc), typeName(tidx), typeName(tdst) });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_diag` (ggml-metal-device.cpp:183
/// @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_diag(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const n: c_int = @intCast(src0.ne[0]);

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_diag_{s}", .{typeName(src0.type)});
    name.set("{s}_n={d}", .{ base.str(), n });

    var res = getOrCompile(lib, &base, &name, null);

    res.nsg = 1;
    res.smem = 0;

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_repeat`
/// (ggml-metal-device.cpp:203 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_repeat(lib: *mc.Library, tsrc: c.enum_ggml_type) callconv(.c) Pwp {
    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_repeat_{s}", .{typeName(tsrc)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_concat`
/// (ggml-metal-device.cpp:218 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_concat(lib: *mc.Library, tsrc: c.enum_ggml_type) callconv(.c) Pwp {
    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_concat_{s}", .{typeName(tsrc)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_unary`
/// (ggml-metal-device.cpp:233 @c1d0e7a00).
///
/// The widest op switch in the file: nine ops plus twenty-two
/// `GGML_UNARY_OP_*`, all onto one kernel specialised by an `op` constant.
pub export fn ggml_metal_library_get_pipeline_unary(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const op_num: c_int = switch (op.op) {
        c.GGML_OP_SCALE => fc.OP_UNARY_NUM_SCALE,
        c.GGML_OP_FILL => fc.OP_UNARY_NUM_FILL,
        c.GGML_OP_CLAMP => fc.OP_UNARY_NUM_CLAMP,
        c.GGML_OP_SQR => fc.OP_UNARY_NUM_SQR,
        c.GGML_OP_SQRT => fc.OP_UNARY_NUM_SQRT,
        c.GGML_OP_SIN => fc.OP_UNARY_NUM_SIN,
        c.GGML_OP_COS => fc.OP_UNARY_NUM_COS,
        c.GGML_OP_LOG => fc.OP_UNARY_NUM_LOG,
        c.GGML_OP_LEAKY_RELU => fc.OP_UNARY_NUM_LEAKY_RELU,
        c.GGML_OP_UNARY => switch (c.ggml_get_unary_op(op)) {
            c.GGML_UNARY_OP_TANH => fc.OP_UNARY_NUM_TANH,
            c.GGML_UNARY_OP_RELU => fc.OP_UNARY_NUM_RELU,
            c.GGML_UNARY_OP_SIGMOID => fc.OP_UNARY_NUM_SIGMOID,
            c.GGML_UNARY_OP_GELU => fc.OP_UNARY_NUM_GELU,
            c.GGML_UNARY_OP_GELU_ERF => fc.OP_UNARY_NUM_GELU_ERF,
            c.GGML_UNARY_OP_GELU_QUICK => fc.OP_UNARY_NUM_GELU_QUICK,
            c.GGML_UNARY_OP_SILU => fc.OP_UNARY_NUM_SILU,
            c.GGML_UNARY_OP_ELU => fc.OP_UNARY_NUM_ELU,
            c.GGML_UNARY_OP_NEG => fc.OP_UNARY_NUM_NEG,
            c.GGML_UNARY_OP_ABS => fc.OP_UNARY_NUM_ABS,
            c.GGML_UNARY_OP_SGN => fc.OP_UNARY_NUM_SGN,
            c.GGML_UNARY_OP_STEP => fc.OP_UNARY_NUM_STEP,
            c.GGML_UNARY_OP_HARDSWISH => fc.OP_UNARY_NUM_HARDSWISH,
            c.GGML_UNARY_OP_HARDSIGMOID => fc.OP_UNARY_NUM_HARDSIGMOID,
            c.GGML_UNARY_OP_EXP => fc.OP_UNARY_NUM_EXP,
            c.GGML_UNARY_OP_SOFTPLUS => fc.OP_UNARY_NUM_SOFTPLUS,
            c.GGML_UNARY_OP_EXPM1 => fc.OP_UNARY_NUM_EXPM1,
            c.GGML_UNARY_OP_FLOOR => fc.OP_UNARY_NUM_FLOOR,
            c.GGML_UNARY_OP_CEIL => fc.OP_UNARY_NUM_CEIL,
            c.GGML_UNARY_OP_ROUND => fc.OP_UNARY_NUM_ROUND,
            c.GGML_UNARY_OP_TRUNC => fc.OP_UNARY_NUM_TRUNC,
            c.GGML_UNARY_OP_XIELU => fc.OP_UNARY_NUM_XIELU,
            else => impl.abort("metal: no unary pipeline for this unary op"),
        },
        else => impl.abort("metal: no unary pipeline for this op"),
    };

    const src0 = src(op, 0);
    const t0_str = typeName(src0.type);
    const t_str = typeName(op.type);

    const is_c4 = @rem(src0.ne[0], 4) == 0;
    const is_cnt = c.ggml_is_contiguous(src0) and c.ggml_nelements(op) < 32768;

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_unary_{s}_{s}{s}", .{ t0_str, t_str, if (is_c4) "_4" else "" });
    name.set("{s}_op={d}_cnt={d}", .{ base.str(), op_num, @as(c_int, if (is_cnt) 1 else 0) });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int16(cv, @intCast(op_num), fc.FC_UNARY + 0);
        mc.ggml_metal_cv_set_bool(cv, is_cnt, fc.FC_UNARY + 1);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    res.c4 = is_c4;
    res.cnt = is_cnt;

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_glu`
/// (ggml-metal-device.cpp:305 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_glu(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    impl.assert(c.ggml_is_contiguous_1(src0), "ggml_is_contiguous_1(op->src[0])");

    const op_str = switch (op.op) {
        c.GGML_OP_GLU => switch (c.ggml_get_glu_op(op)) {
            c.GGML_GLU_OP_REGLU => "reglu",
            c.GGML_GLU_OP_GEGLU => "geglu",
            c.GGML_GLU_OP_SWIGLU => "swiglu",
            c.GGML_GLU_OP_SWIGLU_OAI => "swiglu_oai",
            c.GGML_GLU_OP_GEGLU_ERF => "geglu_erf",
            c.GGML_GLU_OP_GEGLU_QUICK => "geglu_quick",
            else => impl.abort("metal: no glu pipeline for this glu op"),
        },
        else => impl.abort("metal: no glu pipeline for this op"),
    };

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_{s}_{s}", .{ op_str, typeName(src0.type) });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_sum`
/// (ggml-metal-device.cpp:337 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_sum(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_SUM, "op->op == GGML_OP_SUM");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_op_sum_{s}", .{typeName(src(op, 0).type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_sum_rows`
/// (ggml-metal-device.cpp:354 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_sum_rows(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    impl.assert(c.ggml_is_contiguous_rows(src0), "ggml_is_contiguous_rows(op->src[0])");

    const op_num: c_int = switch (op.op) {
        c.GGML_OP_SUM_ROWS => fc.OP_SUM_ROWS_NUM_SUM_ROWS,
        c.GGML_OP_MEAN => fc.OP_SUM_ROWS_NUM_MEAN,
        else => impl.abort("metal: no sum_rows pipeline for this op"),
    };

    const t0_str = typeName(src0.type);
    const t_str = typeName(op.type);

    const is_c4 = @rem(src0.ne[0], 4) == 0;

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_sum_rows_{s}_{s}{s}", .{ t0_str, t_str, if (is_c4) "_4" else "" });
    name.set("{s}_op={d}", .{ base.str(), op_num });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int16(cv, @intCast(op_num), fc.FC_SUM_ROWS + 0);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    res.smem = 32 * @sizeOf(f32);

    if (is_c4) {
        res.smem *= 4;
    }

    res.c4 = is_c4;

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_cumsum_blk`
/// (ggml-metal-device.cpp:398 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_cumsum_blk(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_CUMSUM, "op->op == GGML_OP_CUMSUM");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_cumsum_blk_{s}", .{typeName(src(op, 0).type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_cumsum_add`
/// (ggml-metal-device.cpp:415 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_cumsum_add(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_CUMSUM, "op->op == GGML_OP_CUMSUM");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_cumsum_add_{s}", .{typeName(src(op, 0).type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_tri`
/// (ggml-metal-device.cpp:432 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_tri(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    impl.assert(op.op == c.GGML_OP_TRI, "op->op == GGML_OP_TRI");
    impl.assert(src0.nb[0] == c.ggml_type_size(src0.type), "src0 is contiguous in its rows");

    const op_str = "tri";
    const ttype = op.op_params[0];

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_{s}_{s}_{d}", .{ op_str, typeName(src0.type), @as(c_int, @bitCast(ttype)) });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_soft_max`
/// (ggml-metal-device.cpp:454 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_soft_max(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src1 = srcOpt(op, 1);
    impl.assert(
        src1 == null or src1.?.type == c.GGML_TYPE_F16 or src1.?.type == c.GGML_TYPE_F32,
        "the mask is f16 or f32",
    );

    const suffix: []const u8 = if (@rem(src(op, 0).ne[0], 4) == 0) "_4" else "";
    const tsrc1 = if (src1) |s1| s1.type else c.GGML_TYPE_F32;

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_soft_max_{s}{s}", .{ typeName(tsrc1), suffix });
    name.set("{s}", .{base.str()});

    var res = getOrCompile(lib, &base, &name, null);

    res.smem = 32 * @sizeOf(f32);

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_lightning_indexer`
/// (ggml-metal-device.cpp:481 @c1d0e7a00).
///
/// One of the three that use the same string for `base` and the cache key,
/// passing `name` where the others pass `base`.
pub export fn ggml_metal_library_get_pipeline_lightning_indexer(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_LIGHTNING_INDEXER, "op->op == GGML_OP_LIGHTNING_INDEXER");

    var name: Name = .{};
    name.set("kernel_lightning_indexer_{s}", .{typeName(src(op, 1).type)});

    return getOrCompile(lib, &name, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_dsv4_hc`
/// (ggml-metal-device.cpp:498 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_dsv4_hc(lib: *mc.Library, op: c.enum_ggml_op) callconv(.c) Pwp {
    const kernel = switch (op) {
        c.GGML_OP_DSV4_HC_COMB => "kernel_dsv4_hc_comb_f32",
        c.GGML_OP_DSV4_HC_PRE => "kernel_dsv4_hc_pre_f32",
        c.GGML_OP_DSV4_HC_POST => "kernel_dsv4_hc_post_f32",
        else => impl.abort("metal: no dsv4_hc pipeline for this op"),
    };

    var name: Name = .{};
    name.set("{s}", .{kernel});

    return getOrCompile(lib, &name, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_ssm_conv`
/// (ggml-metal-device.cpp:516 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_ssm_conv(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const src1 = src(op, 1);
    impl.assert(src0.type == c.GGML_TYPE_F32, "op->src[0]->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(op->src[0])");
    impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(op->src[1])");

    const suffix: []const u8 = if (@rem(src1.ne[0], 4) == 0) "_4" else "";

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_ssm_conv_{s}_{s}{s}", .{ typeName(src0.type), typeName(src1.type), suffix });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_ssm_conv_batched`
/// (ggml-metal-device.cpp:543 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_ssm_conv_batched(lib: *mc.Library, op: *const Tensor, ssm_conv_bs: c_int) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const src1 = src(op, 1);
    impl.assert(src0.type == c.GGML_TYPE_F32, "op->src[0]->type == GGML_TYPE_F32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(op->src[0])");
    impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(op->src[1])");

    const suffix: []const u8 = if (@rem(src1.ne[0], 4) == 0) "_4" else "";

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_ssm_conv_{s}_{s}_batched{s}", .{ typeName(src0.type), typeName(src1.type), suffix });
    name.set("{s}_ssm_conv_bs={d}", .{ base.str(), ssm_conv_bs });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int16(cv, @intCast(ssm_conv_bs), fc.FC_SSM_CONV + 0);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_ssm_scan`
/// (ggml-metal-device.cpp:575 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_ssm_scan(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    // The C's `GGML_TENSOR_LOCALS(int32_t, ne0, op->src[0], ne)`, of which
    // only ne00 is used.
    const ne00: i32 = @intCast(src0.ne[0]);

    const nsg = @divTrunc(ne00 + 31, 32);

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_ssm_scan_{s}", .{typeName(src0.type)});
    name.set("{s}_nsg={d}", .{ base.str(), nsg });

    var res = getOrCompile(lib, &base, &name, null);

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (32 + 2) floats
    res.smem = (32 + 2) * @sizeOf(f32) * @as(usize, @intCast(nsg));

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_rwkv`
/// (ggml-metal-device.cpp:601 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_rwkv(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);

    const C = op.ne[0];
    const H = src0.ne[1];

    var base: Name = .{};
    switch (op.op) {
        c.GGML_OP_RWKV_WKV6 => {
            impl.assert(src(op, 5).type == c.GGML_TYPE_F32, "op->src[5]->type == GGML_TYPE_F32");
            impl.assert(@rem(C, H) == 0, "C % H == 0");
            impl.assert(@divTrunc(C, H) == 64, "C / H == 64");

            base.set("kernel_rwkv_wkv6_{s}", .{typeName(src0.type)});
        },
        c.GGML_OP_RWKV_WKV7 => {
            impl.assert(src(op, 6).type == c.GGML_TYPE_F32, "op->src[6]->type == GGML_TYPE_F32");
            impl.assert(@rem(C, H) == 0, "C % H == 0");
            impl.assert(@divTrunc(C, H) == 64, "C / H == 64");

            base.set("kernel_rwkv_wkv7_{s}", .{typeName(src0.type)});
        },
        else => impl.abort("metal: no rwkv pipeline for this op"),
    }

    var name: Name = .{};
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_gated_delta_net`
/// (ggml-metal-device.cpp:639 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_gated_delta_net(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const src2 = src(op, 2);
    const src3 = src(op, 3);

    // v is src[2], dimensions: S_v = ne[0], H = ne[1]
    const ne20: c_int = @intCast(src2.ne[0]); // S_v
    const ne21: c_int = @intCast(src2.ne[1]); // H
    const ne30: c_int = @intCast(src3.ne[0]); // G
    // state is src[5], 4D [S_v, S_v, H_v, n_seqs] (s0 only); K is op param 0.
    const K = impl.getOpParamsI32(op, 0);

    const nsg: c_int = @intCast(@divTrunc(src2.ne[0], 32));

    impl.assert(src(op, 5).type == c.GGML_TYPE_F32, "op->src[5]->type == GGML_TYPE_F32");
    impl.assert(op.ne[0] == @as(i64, ne20) * @as(i64, ne21), "op->ne[0] == ne20 * ne21");
    impl.assert(@rem(ne20, 32) == 0, "ne20 % 32 == 0");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_gated_delta_net_{s}_{d}", .{ typeName(src0.type), nsg });
    name.set("{s}_ne20={d}_ne30={d}_K={d}", .{ base.str(), ne20, ne30, K });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int16(cv, @intCast(ne20), fc.FC_GATED_DELTA_NET + 0);
        mc.ggml_metal_cv_set_int16(cv, @intCast(ne30), fc.FC_GATED_DELTA_NET + 1);
        mc.ggml_metal_cv_set_int16(cv, @intCast(K), fc.FC_GATED_DELTA_NET + 2);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    res.nsg = nsg;

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_solve_tri`
/// (ggml-metal-device.cpp:677 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_solve_tri(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const src1 = src(op, 1);

    const nsg: c_int = 8;
    const n: c_int = @intCast(src1.ne[1]);
    const k: c_int = @intCast(src1.ne[0]);

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_solve_tri_{s}", .{typeName(src0.type)});
    name.set("{s}_nsg={d}_n={d}_k={d}", .{ base.str(), nsg, n, k });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int16(cv, @intCast(nsg), fc.FC_SOLVE_TRI + 0);
        mc.ggml_metal_cv_set_int16(cv, @intCast(n), fc.FC_SOLVE_TRI + 1);
        mc.ggml_metal_cv_set_int16(cv, @intCast(k), fc.FC_SOLVE_TRI + 2);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    res.nsg = nsg;
    res.smem = impl.pad(impl.pad(@intCast(n), 32) * @as(usize, @intCast(nsg)) * @sizeOf(f32), 16);

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_mul_mv_ext`
/// (ggml-metal-device.cpp:707 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_mul_mv_ext(
    lib: *mc.Library,
    op: *const Tensor,
    nsg: c_int,
    nxpsg: c_int,
    r1ptg: c_int,
) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const src1 = src(op, 1);

    const tsrc0 = src0.type;
    const tsrc1 = src1.type;
    const ne12: c_int = @intCast(src1.ne[2]);
    const r2: c_int = @intCast(@divTrunc(src1.ne[2], src0.ne[2]));
    const r3: c_int = @intCast(@divTrunc(src1.ne[3], src0.ne[3]));

    impl.assert(
        ne12 <= std.math.maxInt(i16) and r2 <= std.math.maxInt(i16) and r3 <= std.math.maxInt(i16),
        "ne12, r2 and r3 fit in int16",
    );

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_mul_mv_ext_{s}_{s}_r1_{d}", .{ typeName(tsrc0), typeName(tsrc1), r1ptg });
    name.set("{s}_nsg={d}_nxpsg={d}_ne12={d}_r2={d}_r3={d}", .{ base.str(), nsg, nxpsg, ne12, r2, r3 });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int16(cv, @intCast(nsg), fc.FC_MUL_MV + 0);
        mc.ggml_metal_cv_set_int16(cv, @intCast(nxpsg), fc.FC_MUL_MV + 1);
        mc.ggml_metal_cv_set_int16(cv, @intCast(ne12), fc.FC_MUL_MV + 2);
        mc.ggml_metal_cv_set_int16(cv, @intCast(r2), fc.FC_MUL_MV + 3);
        mc.ggml_metal_cv_set_int16(cv, @intCast(r3), fc.FC_MUL_MV + 4);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_mul_mm`
/// (ggml-metal-device.cpp:740 @c1d0e7a00).
///
/// The tile shape and shared-memory budget depend on whether the device
/// has the tensor instructions, which is why this one asks the device for
/// its props rather than deciding from the tensor alone.
pub export fn ggml_metal_library_get_pipeline_mul_mm(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const src1 = src(op, 1);

    const tsrc0 = src0.type;
    const tsrc1 = src1.type;

    const bc_inp = @rem(src0.ne[0], 32) != 0;

    const NRA: c_int = fc.SZ_SIMDGROUP * fc.N_MM_BLOCK_Y * fc.N_MM_SIMD_GROUP_Y;
    const NRB: c_int = fc.SZ_SIMDGROUP * fc.N_MM_BLOCK_X * fc.N_MM_SIMD_GROUP_X;

    const has_tensor = mc.ggml_metal_device_get_props(mc.ggml_metal_library_get_device(lib)).has_tensor;

    const bc_out = if (has_tensor)
        (@rem(op.ne[0], NRA) != 0 or @rem(op.ne[1], NRB) != 0)
    else
        (@rem(op.ne[0], 64) != 0 or @rem(op.ne[1], 32) != 0);

    impl.assert(
        src1.ne[2] <= std.math.maxInt(i16) and src1.ne[3] <= std.math.maxInt(i16),
        "op->src[1]->ne[2] and ne[3] fit in int16",
    );
    const ne12: i16 = @intCast(src1.ne[2]);
    const ne13: i16 = @intCast(src1.ne[3]);
    const r2: i16 = @intCast(@divTrunc(src1.ne[2], src0.ne[2]));
    const r3: i16 = @intCast(@divTrunc(src1.ne[3], src0.ne[3]));

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_mul_mm_{s}_{s}", .{ typeName(tsrc0), typeName(tsrc1) });
    name.set("{s}_bci={d}_bco={d}_ne12={d}_ne13={d}_r2={d}_r3={d}", .{
        base.str(),
        @as(c_int, if (bc_inp) 1 else 0),
        @as(c_int, if (bc_out) 1 else 0),
        ne12,
        ne13,
        r2,
        r3,
    });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_bool(cv, bc_inp, fc.FC_MUL_MM + 0);
        mc.ggml_metal_cv_set_bool(cv, bc_out, fc.FC_MUL_MM + 1);
        mc.ggml_metal_cv_set_int16(cv, ne12, fc.FC_MUL_MM + 2);
        mc.ggml_metal_cv_set_int16(cv, ne13, fc.FC_MUL_MM + 3);
        mc.ggml_metal_cv_set_int16(cv, r2, fc.FC_MUL_MM + 4);
        mc.ggml_metal_cv_set_int16(cv, r3, fc.FC_MUL_MM + 5);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    if (has_tensor) {
        res.nr0 = NRA;
        res.nr1 = NRB;

        const smem_a: usize = @as(usize, @intCast(NRA)) * @as(usize, @intCast(fc.N_MM_NK_TOTAL)) * @sizeOf(c.ggml_fp16_t);
        res.smem = smem_a;
    } else {
        res.nr0 = 64;
        res.nr1 = 32;

        res.smem = if (bc_out) 8192 else (4096 + 2048);
    }

    res.nsg = fc.N_MM_SIMD_GROUP_X * fc.N_MM_SIMD_GROUP_Y;

    return res;
}

/// The `(nsg, nr0, smem)` a quantized `mul_mv` kernel wants, by weight
/// type -- the quantized arms of
/// `ggml_metal_library_get_pipeline_mul_mv` (ggml-metal-device.cpp:802
/// @c1d0e7a00), which begin at its line 838.
///
/// The C writes 23 `case` blocks assigning two or three locals each; this
/// is the same table. The float types are not here because their arm also
/// chooses a kernel-name suffix from `ne00`, which the caller does.
const MvShape = struct { nsg: c_int, nr0: c_int, smem: usize = 0 };

/// Parameters:
/// - `t`: the weight type to look up.
/// - `log_type`: the type the abort message names. `mul_mv` passes `tsrc0`
///   and `mul_mv_id` passes `op->src[2]->type`; the C really does differ
///   here, and since it only reaches a log line it is reproduced rather
///   than tidied.
fn mvShapeQuant(t: c.enum_ggml_type, log_type: c.enum_ggml_type) MvShape {
    return switch (t) {
        c.GGML_TYPE_Q1_0 => .{ .nsg = fc.N_SG_Q1_0, .nr0 = fc.N_R0_Q1_0 },
        c.GGML_TYPE_Q2_0 => .{ .nsg = fc.N_SG_Q2_0, .nr0 = fc.N_R0_Q2_0 },
        c.GGML_TYPE_Q4_0 => .{ .nsg = fc.N_SG_Q4_0, .nr0 = fc.N_R0_Q4_0 },
        c.GGML_TYPE_Q4_1 => .{ .nsg = fc.N_SG_Q4_1, .nr0 = fc.N_R0_Q4_1 },
        c.GGML_TYPE_Q5_0 => .{ .nsg = fc.N_SG_Q5_0, .nr0 = fc.N_R0_Q5_0 },
        c.GGML_TYPE_Q5_1 => .{ .nsg = fc.N_SG_Q5_1, .nr0 = fc.N_R0_Q5_1 },
        c.GGML_TYPE_Q8_0 => .{ .nsg = fc.N_SG_Q8_0, .nr0 = fc.N_R0_Q8_0, .smem = 32 * @sizeOf(f32) * fc.N_R0_Q8_0 },
        c.GGML_TYPE_MXFP4 => .{ .nsg = fc.N_SG_MXFP4, .nr0 = fc.N_R0_MXFP4, .smem = 32 * @sizeOf(f32) },
        c.GGML_TYPE_Q2_K => .{ .nsg = fc.N_SG_Q2_K, .nr0 = fc.N_R0_Q2_K },
        c.GGML_TYPE_Q3_K => .{ .nsg = fc.N_SG_Q3_K, .nr0 = fc.N_R0_Q3_K },
        c.GGML_TYPE_Q4_K => .{ .nsg = fc.N_SG_Q4_K, .nr0 = fc.N_R0_Q4_K },
        c.GGML_TYPE_Q5_K => .{ .nsg = fc.N_SG_Q5_K, .nr0 = fc.N_R0_Q5_K },
        c.GGML_TYPE_Q6_K => .{ .nsg = fc.N_SG_Q6_K, .nr0 = fc.N_R0_Q6_K },
        c.GGML_TYPE_IQ2_XXS => .{ .nsg = fc.N_SG_IQ2_XXS, .nr0 = fc.N_R0_IQ2_XXS, .smem = 256 * 8 + 128 },
        c.GGML_TYPE_IQ2_XS => .{ .nsg = fc.N_SG_IQ2_XS, .nr0 = fc.N_R0_IQ2_XS, .smem = 512 * 8 + 128 },
        c.GGML_TYPE_IQ3_XXS => .{ .nsg = fc.N_SG_IQ3_XXS, .nr0 = fc.N_R0_IQ3_XXS, .smem = 256 * 4 + 128 },
        c.GGML_TYPE_IQ3_S => .{ .nsg = fc.N_SG_IQ3_S, .nr0 = fc.N_R0_IQ3_S, .smem = 512 * 4 },
        c.GGML_TYPE_IQ2_S => .{ .nsg = fc.N_SG_IQ2_S, .nr0 = fc.N_R0_IQ2_S },
        c.GGML_TYPE_IQ1_S => .{ .nsg = fc.N_SG_IQ1_S, .nr0 = fc.N_R0_IQ1_S },
        c.GGML_TYPE_IQ1_M => .{ .nsg = fc.N_SG_IQ1_M, .nr0 = fc.N_R0_IQ1_M },
        c.GGML_TYPE_IQ4_NL => .{ .nsg = fc.N_SG_IQ4_NL, .nr0 = fc.N_R0_IQ4_NL, .smem = 32 * @sizeOf(f32) },
        c.GGML_TYPE_IQ4_XS => .{ .nsg = fc.N_SG_IQ4_XS, .nr0 = fc.N_R0_IQ4_XS, .smem = 32 * @sizeOf(f32) },
        c.GGML_TYPE_TQ2_0 => .{ .nsg = fc.N_SG_TQ2_0, .nr0 = fc.N_R0_TQ2_0 },
        else => {
            impl.logError("Asserting on type %d\n", .{@as(c_int, @intCast(log_type))});
            impl.abort("metal: mul_mv not implemented for this type");
        },
    };
}

/// Ports `ggml_metal_library_get_pipeline_mul_mv`
/// (ggml-metal-device.cpp:802 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_mul_mv(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const src1 = src(op, 1);

    // The C's two `GGML_TENSOR_LOCALS` blocks, of which these are used.
    const ne00: i32 = @intCast(src0.ne[0]);
    const ne02: i32 = @intCast(src0.ne[2]);
    const ne03: i32 = @intCast(src0.ne[3]);
    const ne12: i32 = @intCast(src1.ne[2]);
    const ne13: i32 = @intCast(src1.ne[3]);

    const tsrc0 = src0.type;
    const tsrc1 = src1.type;

    var nsg: c_int = 0; // number of simdgroups
    var nr0: c_int = 0; // number of src0 rows per simdgroup
    var nr1: c_int = 1; // number of src1 rows per threadgroup
    var smem: usize = 0; // shared memory
    var suffix: []const u8 = "";

    // use custom matrix x vector kernel
    switch (tsrc0) {
        c.GGML_TYPE_F32, c.GGML_TYPE_F16, c.GGML_TYPE_BF16 => {
            if (ne00 < 32) {
                nsg = 1;
                nr0 = 32;
                nr1 = 1;
                suffix = "_short";
            } else {
                nsg = @min(4, @divTrunc(ne00 + 127, 128));
                nr0 = 2;
                nr1 = 1;
                smem = 32 * @sizeOf(f32) * @as(usize, @intCast(nr0));
                suffix = if (@rem(ne00, 4) == 0) "_4" else "";
            }
        },
        else => {
            const shape = mvShapeQuant(tsrc0, tsrc0);
            nsg = shape.nsg;
            nr0 = shape.nr0;
            smem = shape.smem;
        },
    }

    impl.assert(ne12 <= std.math.maxInt(i16) and ne13 <= std.math.maxInt(i16), "ne12 and ne13 fit in int16");
    const r2: i16 = @intCast(@divTrunc(ne12, ne02));
    const r3: i16 = @intCast(@divTrunc(ne13, ne03));

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_mul_mv_{s}_{s}{s}", .{ typeName(tsrc0), typeName(tsrc1), suffix });
    name.set("{s}_nsg={d}_ne12={d}_r2={d}_r3={d}", .{ base.str(), nsg, ne12, r2, r3 });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        // Note the C skips `FC_MUL_MV + 1` here, which `mul_mv_ext` uses
        // for `nxpsg`.
        mc.ggml_metal_cv_set_int16(cv, @intCast(nsg), fc.FC_MUL_MV + 0);
        mc.ggml_metal_cv_set_int16(cv, @intCast(ne12), fc.FC_MUL_MV + 2);
        mc.ggml_metal_cv_set_int16(cv, r2, fc.FC_MUL_MV + 3);
        mc.ggml_metal_cv_set_int16(cv, r3, fc.FC_MUL_MV + 4);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    res.nr0 = nr0;
    res.nr1 = nr1;
    res.nsg = nsg;
    res.smem = smem;

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_mul_mm_id_map0`
/// (ggml-metal-device.cpp:998 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_mul_mm_id_map0(lib: *mc.Library, ne02: c_int, ne20: c_int) callconv(.c) Pwp {
    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_mul_mm_id_map0_ne20_{d}", .{ne20});
    name.set("{s}_ne02={d}", .{ base.str(), ne02 });

    var res = getOrCompile(lib, &base, &name, null);

    res.smem = @as(usize, @intCast(ne02)) * @as(usize, @intCast(ne20)) * @sizeOf(u16);

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_mul_mm_id`
/// (ggml-metal-device.cpp:1015 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_mul_mm_id(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const src1 = src(op, 1);

    const tsrc0 = src0.type;
    const tsrc1 = src1.type;

    const bc_inp = @rem(src0.ne[0], 32) != 0;

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_mul_mm_id_{s}_{s}", .{ typeName(tsrc0), typeName(tsrc1) });
    name.set("{s}_bci={d}", .{ base.str(), @as(c_int, if (bc_inp) 1 else 0) });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_bool(cv, bc_inp, fc.FC_MUL_MM + 0);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    res.smem = 8192;

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_mul_mv_id`
/// (ggml-metal-device.cpp:1043 @c1d0e7a00).
///
/// The quantized arms are `mul_mv`'s, to the value; the float arm differs
/// in having no `_short` path, so `ne00 < 32` is not special-cased.
pub export fn ggml_metal_library_get_pipeline_mul_mv_id(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const src1 = src(op, 1);

    // The C's two `GGML_TENSOR_LOCALS` blocks, of which only ne00 is used.
    const ne00: i32 = @intCast(src0.ne[0]);

    const tsrc0 = src0.type;
    const tsrc1 = src1.type;

    var nsg: c_int = 0; // number of simdgroups
    var nr0: c_int = 0; // number of src0 rows per simdgroup
    var nr1: c_int = 1; // number of src1 rows per threadgroup
    var smem: usize = 0; // shared memory
    var suffix: []const u8 = "";

    // use custom matrix x vector kernel
    switch (tsrc0) {
        c.GGML_TYPE_F32, c.GGML_TYPE_F16, c.GGML_TYPE_BF16 => {
            nsg = @min(4, @divTrunc(ne00 + 127, 128));
            nr0 = 2;
            nr1 = 1;
            smem = 32 * @sizeOf(f32) * @as(usize, @intCast(nr0));
            suffix = if (@rem(ne00, 4) == 0) "_4" else "";
        },
        else => {
            const shape = mvShapeQuant(tsrc0, src(op, 2).type);
            nsg = shape.nsg;
            nr0 = shape.nr0;
            smem = shape.smem;
        },
    }

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_mul_mv_id_{s}_{s}{s}", .{ typeName(tsrc0), typeName(tsrc1), suffix });
    name.set("{s}_nsg={d}", .{ base.str(), nsg });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int16(cv, @intCast(nsg), fc.FC_MUL_MV + 0);
        mc.ggml_metal_cv_set_int16(cv, 1, fc.FC_MUL_MV + 2);
        mc.ggml_metal_cv_set_int16(cv, 1, fc.FC_MUL_MV + 3);
        mc.ggml_metal_cv_set_int16(cv, 1, fc.FC_MUL_MV + 4);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    res.nr0 = nr0;
    res.nr1 = nr1;
    res.nsg = nsg;
    res.smem = smem;

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_argmax`
/// (ggml-metal-device.cpp:1228 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_argmax(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    const src0 = src(op, 0);
    impl.assert(src0.type == c.GGML_TYPE_F32, "op->src[0]->type == GGML_TYPE_F32");
    impl.assert(c.ggml_is_contiguous_1(src0), "ggml_is_contiguous_1(op->src[0])");
    impl.assert(src0.nb[0] == c.ggml_type_size(src0.type), "src0 is contiguous in its rows");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_argmax_{s}", .{typeName(src0.type)});
    name.set("{s}", .{base.str()});

    var res = getOrCompile(lib, &base, &name, null);

    res.smem = 32 * (@sizeOf(f32) + @sizeOf(i32));

    return res;
}

/// The `ggml_sort_order` spelling the four argsort kernel names carry --
/// the switch inside `ggml_metal_library_get_pipeline_argsort`
/// (ggml-metal-device.cpp:1249 @c1d0e7a00) and its three twins.
///
/// The C repeats this switch four times, `"undefined"` initialiser and
/// all. The initialiser is dead in every copy -- the `default` aborts --
/// so it is not reproduced.
fn sortOrderName(order: c.enum_ggml_sort_order) []const u8 {
    return switch (order) {
        c.GGML_SORT_ORDER_ASC => "asc",
        c.GGML_SORT_ORDER_DESC => "desc",
        else => impl.abort("metal: unknown sort order"),
    };
}

/// Ports `ggml_metal_library_get_pipeline_argsort`
/// (ggml-metal-device.cpp:1249 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_argsort(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_ARGSORT, "op->op == GGML_OP_ARGSORT");

    const order: c.enum_ggml_sort_order = @intCast(op.op_params[0]);
    const order_str = sortOrderName(order);

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_argsort_{s}_{s}_{s}", .{ typeName(src(op, 0).type), typeName(op.type), order_str });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_argsort_merge`
/// (ggml-metal-device.cpp:1275 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_argsort_merge(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_ARGSORT, "op->op == GGML_OP_ARGSORT");

    const order: c.enum_ggml_sort_order = @intCast(op.op_params[0]);
    const order_str = sortOrderName(order);

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_argsort_merge_{s}_{s}_{s}", .{ typeName(src(op, 0).type), typeName(op.type), order_str });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_fwht`
/// (ggml-metal-device.cpp:1301 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_fwht(lib: *mc.Library, n: c_int) callconv(.c) Pwp {
    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_fwht_f32_{d}", .{n});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_top_k`
/// (ggml-metal-device.cpp:1317 @c1d0e7a00).
///
/// note: reuse the argsort kernel for top_k
pub export fn ggml_metal_library_get_pipeline_top_k(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_TOP_K, "op->op == GGML_OP_TOP_K");

    // note: the top_k kernel is always descending order
    const order_str = sortOrderName(c.GGML_SORT_ORDER_DESC);

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_argsort_{s}_{s}_{s}", .{ typeName(src(op, 0).type), typeName(op.type), order_str });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_top_k_merge`
/// (ggml-metal-device.cpp:1344 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_top_k_merge(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_TOP_K, "op->op == GGML_OP_TOP_K");

    const order_str = sortOrderName(c.GGML_SORT_ORDER_DESC);

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_argsort_merge_{s}_{s}_{s}", .{ typeName(src(op, 0).type), typeName(op.type), order_str });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_flash_attn_ext_pad`
/// (ggml-metal-device.cpp:1370 @c1d0e7a00).
///
/// The C leaves eight `ggml_metal_cv_set_*` calls commented out here and
/// in `_blk` below, keeping the constant indices they would use. Only the
/// live ones are ported; the indices they reserve are visible in
/// `impl_c.zig`.
pub export fn ggml_metal_library_get_pipeline_flash_attn_ext_pad(
    lib: *mc.Library,
    op: *const Tensor,
    has_mask: bool,
    ncpsg: i32,
) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_{s}", .{"flash_attn_ext_pad"});
    name.set("{s}_mask={d}_ncpsg={d}", .{ base.str(), @as(c_int, if (has_mask) 1 else 0), ncpsg });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_bool(cv, has_mask, fc.FC_FLASH_ATTN_EXT_PAD + 0);

        mc.ggml_metal_cv_set_int32(cv, ncpsg, fc.FC_FLASH_ATTN_EXT_PAD + 25);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16`
/// (ggml-metal-device.cpp:1413 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    var base: Name = .{};
    base.set("kernel_flash_attn_ext_kv_{s}_f16", .{typeName(src(op, 1).type)});

    return getOrCompile(lib, &base, &base, null);
}

/// Ports `ggml_metal_library_get_pipeline_flash_attn_ext_blk`
/// (ggml-metal-device.cpp:1430 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_flash_attn_ext_blk(
    lib: *mc.Library,
    op: *const Tensor,
    nqptg: i32,
    ncpsg: i32,
) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_{s}", .{"flash_attn_ext_blk"});
    name.set("{s}_nqptg={d}_ncpsg={d}", .{ base.str(), nqptg, ncpsg });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int32(cv, nqptg, fc.FC_FLASH_ATTN_EXT_BLK + 24);
        mc.ggml_metal_cv_set_int32(cv, ncpsg, fc.FC_FLASH_ATTN_EXT_BLK + 25);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_flash_attn_ext`
/// (ggml-metal-device.cpp:1473 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_flash_attn_ext(
    lib: *mc.Library,
    op: *const Tensor,
    has_mask: bool,
    has_sinks: bool,
    has_bias: bool,
    has_scap: bool,
    has_kvpad: bool,
    nsg: i32,
    use_kv_f16: bool,
    ns10: i32,
    ns20: i32,
) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const dk: i32 = @intCast(src(op, 1).ne[0]);
    const dv: i32 = @intCast(src(op, 2).ne[0]);

    const kv_type = if (use_kv_f16) "f16" else typeName(src(op, 1).type);

    // do bounds checks for the mask?
    const src3 = srcOpt(op, 3);
    const bc_mask = src3 != null and @rem(src3.?.ne[1], 8) != 0;

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_{s}_{s}_dk{d}_dv{d}", .{ "flash_attn_ext", kv_type, dk, dv });
    name.set("{s}_mask={d}_sinks={d}_bias={d}_scap={d}_kvpad={d}_bcm={d}_ns10={d}_ns20={d}_nsg={d}", .{
        base.str(),
        @as(c_int, if (has_mask) 1 else 0),
        @as(c_int, if (has_sinks) 1 else 0),
        @as(c_int, if (has_bias) 1 else 0),
        @as(c_int, if (has_scap) 1 else 0),
        @as(c_int, if (has_kvpad) 1 else 0),
        @as(c_int, if (bc_mask) 1 else 0),
        ns10,
        ns20,
        nsg,
    });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_bool(cv, has_mask, fc.FC_FLASH_ATTN_EXT + 0);
        mc.ggml_metal_cv_set_bool(cv, has_sinks, fc.FC_FLASH_ATTN_EXT + 1);
        mc.ggml_metal_cv_set_bool(cv, has_bias, fc.FC_FLASH_ATTN_EXT + 2);
        mc.ggml_metal_cv_set_bool(cv, has_scap, fc.FC_FLASH_ATTN_EXT + 3);
        mc.ggml_metal_cv_set_bool(cv, has_kvpad, fc.FC_FLASH_ATTN_EXT + 4);

        mc.ggml_metal_cv_set_bool(cv, bc_mask, fc.FC_FLASH_ATTN_EXT + 10);

        mc.ggml_metal_cv_set_int32(cv, ns10, fc.FC_FLASH_ATTN_EXT + 20);
        mc.ggml_metal_cv_set_int32(cv, ns20, fc.FC_FLASH_ATTN_EXT + 21);
        mc.ggml_metal_cv_set_int32(cv, nsg, fc.FC_FLASH_ATTN_EXT + 22);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_flash_attn_ext_vec`
/// (ggml-metal-device.cpp:1540 @c1d0e7a00).
///
/// The only selection function that consults the tuning table: a
/// `(nqpsg, ne)` pair equal to the baseline for this `(dk, dv)` is left
/// out of the kernel name, so the generic kernel is reused rather than a
/// specialisation compiled per shape.
pub export fn ggml_metal_library_get_pipeline_flash_attn_ext_vec(
    lib: *mc.Library,
    op: *const Tensor,
    has_mask: bool,
    has_sinks: bool,
    has_bias: bool,
    has_scap: bool,
    has_kvpad: bool,
    nqpsg: i32,
    ne: i32,
    nsg: i32,
    nwg: i32,
    use_kv_f16: bool,
    ns10: i32,
    ns20: i32,
) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    const dk: i32 = @intCast(src(op, 1).ne[0]);
    const dv: i32 = @intCast(src(op, 2).ne[0]);

    const kv_type = if (use_kv_f16) "f16" else typeName(src(op, 1).type);

    // The C's `char qne_suffix[16] = {0}`, written by `snprintf` with that
    // size -- so it truncates rather than overflows, and the truncation is
    // kept.
    var qne_buf: [64]u8 = undefined;
    var qne_suffix: []const u8 = "";
    if (!(nqpsg == 1 and ne == tuning.baselineNe(dk, dv))) {
        const out = std.fmt.bufPrint(&qne_buf, "_q{d}_ne{d}", .{ nqpsg, ne }) catch
            impl.abort("metal: flash_attn_ext_vec suffix does not fit");
        qne_suffix = out[0..@min(out.len, 15)];
    }

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_{s}_{s}_dk{d}_dv{d}{s}", .{ "flash_attn_ext_vec", kv_type, dk, dv, qne_suffix });
    name.set("{s}_mask={d}_sink={d}_bias={d}_scap={d}_kvpad={d}_ns10={d}_ns20={d}_nsg={d}_nwg={d}", .{
        base.str(),
        @as(c_int, if (has_mask) 1 else 0),
        @as(c_int, if (has_sinks) 1 else 0),
        @as(c_int, if (has_bias) 1 else 0),
        @as(c_int, if (has_scap) 1 else 0),
        @as(c_int, if (has_kvpad) 1 else 0),
        ns10,
        ns20,
        nsg,
        nwg,
    });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_bool(cv, has_mask, fc.FC_FLASH_ATTN_EXT_VEC + 0);
        mc.ggml_metal_cv_set_bool(cv, has_sinks, fc.FC_FLASH_ATTN_EXT_VEC + 1);
        mc.ggml_metal_cv_set_bool(cv, has_bias, fc.FC_FLASH_ATTN_EXT_VEC + 2);
        mc.ggml_metal_cv_set_bool(cv, has_scap, fc.FC_FLASH_ATTN_EXT_VEC + 3);
        mc.ggml_metal_cv_set_bool(cv, has_kvpad, fc.FC_FLASH_ATTN_EXT_VEC + 4);

        mc.ggml_metal_cv_set_int32(cv, ns10, fc.FC_FLASH_ATTN_EXT_VEC + 20);
        mc.ggml_metal_cv_set_int32(cv, ns20, fc.FC_FLASH_ATTN_EXT_VEC + 21);
        mc.ggml_metal_cv_set_int32(cv, nsg, fc.FC_FLASH_ATTN_EXT_VEC + 22);
        mc.ggml_metal_cv_set_int32(cv, nwg, fc.FC_FLASH_ATTN_EXT_VEC + 23);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce`
/// (ggml-metal-device.cpp:1611 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
    lib: *mc.Library,
    op: *const Tensor,
    dv: i32,
    nwg: i32,
) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_FLASH_ATTN_EXT, "op->op == GGML_OP_FLASH_ATTN_EXT");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_flash_attn_ext_vec_reduce", .{});
    name.set("{s}_dv={d}_nwg={d}", .{ base.str(), dv, nwg });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int32(cv, dv, fc.FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        mc.ggml_metal_cv_set_int32(cv, nwg, fc.FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    return res;
}

/// The `op` constant the four `bin` ops share -- the switch inside
/// `ggml_metal_library_get_pipeline_bin` (ggml-metal-device.cpp:1641
/// @c1d0e7a00) and its twin in `_bin_one`.
///
/// These four are literals in the C, not named constants, and they are the
/// same switch in `_bin` and `_bin_one`.
fn binOpNum(op: c.enum_ggml_op) c_int {
    return switch (op) {
        c.GGML_OP_ADD => 0,
        c.GGML_OP_SUB => 1,
        c.GGML_OP_MUL => 2,
        c.GGML_OP_DIV => 3,
        else => impl.abort("metal: no bin pipeline for this op"),
    };
}

/// Ports `ggml_metal_library_get_pipeline_bin`
/// (ggml-metal-device.cpp:1641 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_bin(lib: *mc.Library, op: *const Tensor, n_fuse: i32) callconv(.c) Pwp {
    const src0 = src(op, 0);
    const src1 = src(op, 1);

    const op_num = binOpNum(op.op);

    const t0_str = typeName(src0.type);
    const t1_str = typeName(src1.type);
    const t_str = typeName(op.type);

    const is_c4 = (@rem(src0.ne[0], 4) == 0) and (@rem(src1.ne[0], 4) == 0);

    const is_cb = src0.ne[0] != src1.ne[0];
    const is_rb = c.ggml_is_contiguous(src0) and c.ggml_is_contiguous(src1) and
        (c.ggml_nrows(src1) == 1) and c.ggml_nelements(op) < 65536;

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_bin_fuse_{s}_{s}_{s}{s}", .{ t0_str, t1_str, t_str, if (is_c4) "_4" else "" });
    name.set("{s}_op={d}_nf={d}_rb={d}_cb={d}", .{
        base.str(),
        op_num,
        n_fuse,
        @as(c_int, if (is_rb) 1 else 0),
        @as(c_int, if (is_cb) 1 else 0),
    });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int16(cv, @intCast(op_num), fc.FC_BIN + 0);
        mc.ggml_metal_cv_set_int16(cv, @intCast(n_fuse), fc.FC_BIN + 1);
        mc.ggml_metal_cv_set_bool(cv, is_rb, fc.FC_BIN + 2);
        mc.ggml_metal_cv_set_bool(cv, is_cb, fc.FC_BIN + 3);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    res.c4 = is_c4;
    res.cnt = is_rb;

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_bin_one`
/// (ggml-metal-device.cpp:1687 @c1d0e7a00).
///
/// Note the C sets only three of the four `FC_BIN` constants here, leaving
/// `+ 3` (`is_cb`) at the kernel's own default.
pub export fn ggml_metal_library_get_pipeline_bin_one(lib: *mc.Library, op: c.enum_ggml_op) callconv(.c) Pwp {
    const op_num = binOpNum(op);

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_bin_fuse_{s}_{s}_{s}", .{ "f32", "f32", "f32" });
    name.set("{s}_op={d}_nf={d}", .{ base.str(), op_num, @as(c_int, 1) });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int16(cv, @intCast(op_num), fc.FC_BIN + 0);
        mc.ggml_metal_cv_set_int16(cv, 1, fc.FC_BIN + 1);
        mc.ggml_metal_cv_set_bool(cv, false, fc.FC_BIN + 2);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_l2_norm`
/// (ggml-metal-device.cpp:1720 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_l2_norm(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_L2_NORM, "op->op == GGML_OP_L2_NORM");

    const src0 = src(op, 0);
    const is_c4 = @rem(src0.ne[0], 4) == 0;

    const t0_str = typeName(src0.type);
    const t_str = typeName(op.type);

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_l2_norm_{s}_{s}{s}", .{ t0_str, t_str, if (is_c4) "_4" else "" });
    name.set("{s}", .{base.str()});

    var res = getOrCompile(lib, &base, &name, null);

    res.c4 = is_c4;
    res.smem = 32 * @sizeOf(f32);

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_group_norm`
/// (ggml-metal-device.cpp:1745 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_group_norm(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_GROUP_NORM, "op->op == GGML_OP_GROUP_NORM");
    impl.assert(c.ggml_is_contiguous(src(op, 0)), "ggml_is_contiguous(op->src[0])");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_group_norm_f32", .{});
    name.set("{s}", .{base.str()});

    var res = getOrCompile(lib, &base, &name, null);

    res.smem = 32 * @sizeOf(f32);

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_norm`
/// (ggml-metal-device.cpp:1766 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_norm(lib: *mc.Library, op: *const Tensor, n_fuse: c_int) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_NORM or op.op == c.GGML_OP_RMS_NORM, "op->op is NORM or RMS_NORM");
    impl.assert(c.ggml_is_contiguous_rows(src(op, 0)), "ggml_is_contiguous_rows(op->src[0])");

    const suffix: []const u8 = if (@rem(op.ne[0], 4) == 0) "_4" else "";

    var base: Name = .{};
    switch (op.op) {
        c.GGML_OP_NORM => switch (n_fuse) {
            1 => base.set("kernel_norm_f32{s}", .{suffix}),
            2 => base.set("kernel_norm_mul_f32{s}", .{suffix}),
            3 => base.set("kernel_norm_mul_add_f32{s}", .{suffix}),
            else => impl.abort("metal: no norm pipeline for this fusion count"),
        },
        c.GGML_OP_RMS_NORM => switch (n_fuse) {
            1 => base.set("kernel_rms_norm_f32{s}", .{suffix}),
            2 => base.set("kernel_rms_norm_mul_f32{s}", .{suffix}),
            3 => base.set("kernel_rms_norm_mul_add_f32{s}", .{suffix}),
            else => impl.abort("metal: no rms_norm pipeline for this fusion count"),
        },
        else => impl.abort("metal: no norm pipeline for this op"),
    }

    var name: Name = .{};
    name.set("{s}", .{base.str()});

    var res = getOrCompile(lib, &base, &name, null);

    res.smem = 32 * @sizeOf(f32);

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_rope`
/// (ggml-metal-device.cpp:1809 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_rope(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_ROPE or op.op == c.GGML_OP_ROPE_BACK, "op->op is ROPE or ROPE_BACK");

    const is_back = op.op == c.GGML_OP_ROPE_BACK;

    const src0 = src(op, 0);

    const mode = impl.getOpParamsI32(op, 2);

    const is_neox = (mode & c.GGML_ROPE_TYPE_NEOX) != 0;
    const is_mrope = (mode & c.GGML_ROPE_TYPE_MROPE) != 0;
    const is_imrope = mode == c.GGML_ROPE_TYPE_IMROPE;
    const is_vision = mode == c.GGML_ROPE_TYPE_VISION;

    var base: Name = .{};
    if (is_neox) {
        base.set("kernel_rope_neox_{s}", .{typeName(src0.type)});
    } else if ((is_mrope or is_imrope) and !is_vision) {
        // need at least 4 pos per token
        impl.assert(src(op, 1).ne[0] * 4 >= src0.ne[2], "op->src[1]->ne[0]*4 >= op->src[0]->ne[2]");
        base.set("kernel_rope_multi_{s}", .{typeName(src0.type)});
    } else if (is_vision) {
        // need at least 4 pos per token
        impl.assert(src(op, 1).ne[0] * 4 >= src0.ne[2], "op->src[1]->ne[0]*4 >= op->src[0]->ne[2]");
        base.set("kernel_rope_vision_{s}", .{typeName(src0.type)});
    } else {
        base.set("kernel_rope_norm_{s}", .{typeName(src0.type)});
    }

    var name: Name = .{};
    name.set("{s}_imrope={d}_is_back={d}", .{
        base.str(),
        @as(c_int, if (is_imrope) 1 else 0),
        @as(c_int, if (is_back) 1 else 0),
    });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_bool(cv, is_imrope, fc.FC_ROPE + 0);
        mc.ggml_metal_cv_set_bool(cv, is_back, fc.FC_ROPE + 1);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_im2col`
/// (ggml-metal-device.cpp:1853 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_im2col(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_IM2COL, "op->op == GGML_OP_IM2COL");

    const src0 = src(op, 0);
    // The C's `GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne)`.
    const ne00 = src0.ne[0];
    const ne01 = src0.ne[1];

    impl.assert(c.ggml_is_contiguous(src(op, 1)), "ggml_is_contiguous(op->src[1])");
    impl.assert(src(op, 1).type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(op.type == c.GGML_TYPE_F16 or op.type == c.GGML_TYPE_F32, "op->type is f16 or f32");

    const is_2D = impl.getOpParamsI32(op, 6) == 1;
    const KH: i64 = if (is_2D) ne01 else 1;
    const KW: i64 = ne00;

    var base: Name = .{};
    var name: Name = .{};
    if (KH * KW <= 1024) {
        base.set("kernel_im2col_{s}", .{typeName(op.type)});
    } else {
        base.set("kernel_im2col_ext_{s}", .{typeName(op.type)});
    }
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_conv_transpose_1d`
/// (ggml-metal-device.cpp:1884 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_conv_transpose_1d(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_CONV_TRANSPOSE_1D, "op->op == GGML_OP_CONV_TRANSPOSE_1D");

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(op->src[0])");
    impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(op->src[1])");
    impl.assert(src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_F32, "op->src[0]->type is f16 or f32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_conv_transpose_1d_{s}_{s}", .{ typeName(src0.type), typeName(src1.type) });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_col2im_1d`
/// (ggml-metal-device.cpp:1907 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_col2im_1d(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_COL2IM_1D, "op->op == GGML_OP_COL2IM_1D");

    const src0 = src(op, 0);
    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(op->src[0])");
    impl.assert(
        src0.type == c.GGML_TYPE_F32 or src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_BF16,
        "op->src[0]->type is f32, f16 or bf16",
    );

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_col2im_1d_{s}", .{typeName(src0.type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_snake`
/// (ggml-metal-device.cpp:1927 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_snake(lib: *mc.Library, @"type": c.enum_ggml_type) callconv(.c) Pwp {
    impl.assert(
        @"type" == c.GGML_TYPE_F32 or @"type" == c.GGML_TYPE_F16 or @"type" == c.GGML_TYPE_BF16,
        "type is f32, f16 or bf16",
    );

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_snake_{s}", .{typeName(@"type")});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_conv_transpose_2d`
/// (ggml-metal-device.cpp:1944 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_conv_transpose_2d(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_CONV_TRANSPOSE_2D, "op->op == GGML_OP_CONV_TRANSPOSE_2D");

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(op->src[0])");
    impl.assert(c.ggml_is_contiguous(src1), "ggml_is_contiguous(op->src[1])");
    impl.assert(src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_F32, "op->src[0]->type is f16 or f32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_conv_transpose_2d_{s}_{s}", .{ typeName(src0.type), typeName(src1.type) });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_conv_2d`
/// (ggml-metal-device.cpp:1967 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_conv_2d(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_CONV_2D, "op->op == GGML_OP_CONV_2D");

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(op->src[0])");
    impl.assert(src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_F32, "op->src[0]->type is f16 or f32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_conv_2d_{s}_{s}", .{ typeName(src0.type), typeName(src1.type) });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_conv_2d_dw`
/// (ggml-metal-device.cpp:1989 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_conv_2d_dw(lib: *mc.Library, op: *const Tensor, tiled: bool) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_CONV_2D_DW, "op->op == GGML_OP_CONV_2D_DW");

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    impl.assert(src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_F32, "op->src[0]->type is f16 or f32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_conv_2d_dw{s}_{s}_{s}", .{
        if (tiled) "_tiled" else "",
        typeName(src0.type),
        typeName(src1.type),
    });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_conv_3d`
/// (ggml-metal-device.cpp:2012 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_conv_3d(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_CONV_3D, "op->op == GGML_OP_CONV_3D");

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    impl.assert(c.ggml_is_contiguous(src0), "ggml_is_contiguous(op->src[0])");
    impl.assert(src0.type == c.GGML_TYPE_F16 or src0.type == c.GGML_TYPE_F32, "op->src[0]->type is f16 or f32");
    impl.assert(src1.type == c.GGML_TYPE_F32, "op->src[1]->type == GGML_TYPE_F32");
    impl.assert(op.type == c.GGML_TYPE_F32, "op->type == GGML_TYPE_F32");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_conv_3d_{s}_{s}", .{ typeName(src0.type), typeName(src1.type) });
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_upscale`
/// (ggml-metal-device.cpp:2034 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_upscale(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_UPSCALE, "op->op == GGML_OP_UPSCALE");

    const src0 = src(op, 0);

    const mode_flags = impl.getOpParamsI32(op, 0);
    const mode: c.enum_ggml_scale_mode = @intCast(mode_flags & 0xFF);

    const antialias = (mode_flags & c.GGML_SCALE_FLAG_ANTIALIAS) != 0;

    var base: Name = .{};
    if (mode == c.GGML_SCALE_MODE_BILINEAR) {
        base.set("kernel_upscale_bilinear_{s}", .{typeName(src0.type)});
    } else if (mode == c.GGML_SCALE_MODE_BICUBIC) {
        base.set("kernel_upscale_bicubic_{s}", .{typeName(src0.type)});
    } else {
        base.set("kernel_upscale_nearest_{s}", .{typeName(src0.type)});
    }

    var name: Name = .{};
    name.set("{s}_aa={d}", .{ base.str(), @as(c_int, if (antialias) 1 else 0) });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_bool(cv, antialias, fc.FC_UPSCALE + 0);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_roll`
/// (ggml-metal-device.cpp:2068 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_roll(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_ROLL, "op->op == GGML_OP_ROLL");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_roll_{s}", .{typeName(src(op, 0).type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_pad`
/// (ggml-metal-device.cpp:2085 @c1d0e7a00).
///
/// The one function that does not use the shared `getOrCompile` shape: a
/// cache hit returns before `res.c4` is set, so the field is written only
/// on the compile path. `is_c4` is hard-`false` here -- the C keeps the
/// expression that would compute it commented out, noting it is slower --
/// so the two paths agree today, but the asymmetry is the C's and is kept.
pub export fn ggml_metal_library_get_pipeline_pad(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_PAD, "op->op == GGML_OP_PAD");

    // note: this is slower
    //const is_c4 = @rem(src(op, 0).ne[0], 4) == 0 and @rem(op.ne[0], 4) == 0;
    const is_c4 = false;

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_pad_{s}{s}", .{ typeName(src(op, 0).type), if (is_c4) "_4" else "" });
    name.set("{s}", .{base.str()});

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline != null) {
        return res;
    }

    res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), null);

    res.c4 = is_c4;

    return res;
}

/// Ports `ggml_metal_library_get_pipeline_pad_reflect_1d`
/// (ggml-metal-device.cpp:2110 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_pad_reflect_1d(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_PAD_REFLECT_1D, "op->op == GGML_OP_PAD_REFLECT_1D");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_pad_reflect_1d_{s}", .{typeName(src(op, 0).type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_arange`
/// (ggml-metal-device.cpp:2127 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_arange(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_ARANGE, "op->op == GGML_OP_ARANGE");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_arange_{s}", .{typeName(op.type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_timestep_embedding`
/// (ggml-metal-device.cpp:2144 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_timestep_embedding(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_TIMESTEP_EMBEDDING, "op->op == GGML_OP_TIMESTEP_EMBEDDING");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_timestep_embedding_{s}", .{typeName(src(op, 0).type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_opt_step_adamw`
/// (ggml-metal-device.cpp:2161 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_opt_step_adamw(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_OPT_STEP_ADAMW, "op->op == GGML_OP_OPT_STEP_ADAMW");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_opt_step_adamw_{s}", .{typeName(src(op, 0).type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_opt_step_sgd`
/// (ggml-metal-device.cpp:2178 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_opt_step_sgd(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_OPT_STEP_SGD, "op->op == GGML_OP_OPT_STEP_SGD");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_opt_step_sgd_{s}", .{typeName(src(op, 0).type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_silu_back`
/// (ggml-metal-device.cpp:2195 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_silu_back(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_SILU_BACK, "op->op == GGML_OP_SILU_BACK");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_silu_back_{s}", .{typeName(src(op, 0).type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_memset`
/// (ggml-metal-device.cpp:2212 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_memset(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.type == c.GGML_TYPE_I64, "op->type == GGML_TYPE_I64");

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_memset_{s}", .{typeName(op.type)});
    name.set("{s}", .{base.str()});

    return getOrCompile(lib, &base, &name, null);
}

/// Ports `ggml_metal_library_get_pipeline_count_equal`
/// (ggml-metal-device.cpp:2229 @c1d0e7a00).
pub export fn ggml_metal_library_get_pipeline_count_equal(lib: *mc.Library, op: *const Tensor) callconv(.c) Pwp {
    impl.assert(op.op == c.GGML_OP_COUNT_EQUAL, "op->op == GGML_OP_COUNT_EQUAL");

    const src0 = src(op, 0);
    const src1 = src(op, 1);

    // The C's `GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne)`.
    const ne00 = src0.ne[0];

    impl.assert(src0.type == src1.type, "op->src[0]->type == op->src[1]->type");
    impl.assert(src0.type == c.GGML_TYPE_I32, "op->src[0]->type == GGML_TYPE_I32");
    impl.assert(op.type == c.GGML_TYPE_I64, "op->type == GGML_TYPE_I64");

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    impl.assert(c.ggml_nelements(src0) < (@as(i64, 1) << 31), "ggml_nelements(op->src[0]) < (1LL << 31)");

    var nsg: c_int = 1;
    while (32 * nsg < ne00 and nsg < 32) {
        nsg *= 2;
    }

    var base: Name = .{};
    var name: Name = .{};
    base.set("kernel_count_equal_{s}", .{typeName(src0.type)});
    name.set("{s}_nsg={d}", .{ base.str(), nsg });

    var res = mc.ggml_metal_library_get_pipeline(lib, name.z());
    if (res.pipeline == null) {
        const cv = mc.ggml_metal_cv_init().?;
        defer mc.ggml_metal_cv_free(cv);

        mc.ggml_metal_cv_set_int16(cv, @intCast(nsg), fc.FC_COUNT_EQUAL + 0);

        res = mc.ggml_metal_library_compile_pipeline(lib, base.z(), name.z(), cv);
    }

    res.smem = 32 * @sizeOf(i32);
    res.nsg = nsg;

    return res;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the pipeline cache copies its keys" {
    // The caller's `name` is a stack buffer in every call site, so a cache
    // that borrowed the key would read freed memory on the next lookup.
    //
    // The handles are real, from the `.m`'s own `pipeline_init`, because
    // `ggml_metal_pipelines_free` frees every value it holds -- as the C
    // does. Fake pointers crashed here the moment this file joined the
    // build and was linked against the `.m` for real.
    const ppls = ggml_metal_pipelines_init().?;
    defer ggml_metal_pipelines_free(ppls);

    const fake = mc.ggml_metal_pipeline_init().?;
    {
        var scratch: [32]u8 = undefined;
        const key = try std.fmt.bufPrintZ(&scratch, "kernel_{s}", .{"abc"});
        ggml_metal_pipelines_add(ppls, key, fake);
        @memset(&scratch, 0xaa); // the caller's buffer goes away
    }
    try std.testing.expectEqual(fake, ggml_metal_pipelines_get(ppls, "kernel_abc").?);
    try std.testing.expectEqual(@as(?*mc.Pipeline, null), ggml_metal_pipelines_get(ppls, "kernel_xyz"));
}

test "adding the same name twice replaces rather than duplicating" {
    const ppls = ggml_metal_pipelines_init().?;
    defer ggml_metal_pipelines_free(ppls);

    const a = mc.ggml_metal_pipeline_init().?;
    const b = mc.ggml_metal_pipeline_init().?;
    ggml_metal_pipelines_add(ppls, "k", a);
    ggml_metal_pipelines_add(ppls, "k", b);
    // `add` replaces without freeing, as the C does, so `a` is now
    // unreachable from the cache and this test frees it by hand.
    mc.ggml_metal_pipeline_free(a);
    try std.testing.expectEqual(b, ggml_metal_pipelines_get(ppls, "k").?);

    const self: *Pipelines = @ptrCast(@alignCast(ppls));
    try std.testing.expectEqual(@as(u32, 1), self.data.count());
}

test "a name longer than the C's 256 aborts rather than truncating" {
    // Not runnable -- `Name.set` aborts -- so what is checked is that the
    // buffer is the C's size, since that is what the bound depends on.
    var n: Name = .{};
    n.set("kernel_{s}", .{"cpy_f32_f16"});
    try std.testing.expectEqualStrings("kernel_cpy_f32_f16", n.str());
    try std.testing.expectEqual(@as(u8, 0), n.buf[n.len]);
    try std.testing.expectEqual(@as(usize, 256), n.buf.len);
}
