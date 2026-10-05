//! The Metal host layer's internal C surface, hand-declared.
//!
//! # Provenance
//!
//! Mirrors the declarations `ggml-metal.cpp` uses from, in the reference
//! checkout:
//!
//! - `llama.cpp/ggml/src/ggml-metal/ggml-metal-device.h`
//! - `llama.cpp/ggml/src/ggml-metal/ggml-metal-context.h`
//! - `llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.h`
//!
//! all at v0.3.0 (`c1d0e7a00`).
//!
//! # Why these are hand-written rather than imported
//!
//! All three headers **do** import cleanly — measured, not assumed; they
//! are `extern "C"` and include only `ggml.h`, with the Objective-C behind
//! them in the `.m` files. Putting them in `impl.zig`'s `cImport` was
//! tried and works.
//!
//! It was reverted deliberately. That `cImport` is shared by every ported
//! file, so adding the Metal headers to it widens the `c` namespace the
//! whole library sees and obliges six build and script sites to carry a
//! new include path. A second `cImport` inside `metal/` is not an option:
//! two of them produce two incompatible `*ggml_tensor`, and the error says
//! only that `*cimport.struct_ggml_tensor` will not coerce to
//! `*cimport.struct_ggml_tensor` — the trap `cli/c.zig` records for
//! `llama.h`. So the surface this one directory needs is declared here,
//! and nothing else changes.
//!
//! # The struct layout is checked, not trusted
//!
//! `CLAUDE.md` calls hand-transcription "the single largest source of
//! silent error", and `DeviceProps` below is 19 fields of which this port
//! reads seven — the other twelve exist only to place those seven at the
//! right offsets. Getting one wrong reads an adjacent field and reports
//! nothing.
//!
//! **`make struct-layout` is the gate.** It compiles the real header and
//! compares `sizeof` and every `offsetof` against the values this file
//! exports. Negative-tested by perturbing a field.

const std = @import("std");
const impl = @import("../impl.zig");

const c = impl.c;

pub const Tensor = c.ggml_tensor;
pub const CGraph = c.ggml_cgraph;

/// Mirrors `enum ggml_metal_device_id` (ggml-metal-device.h:238
/// @c1d0e7a00).
pub const DeviceId = @import("tuning_table.zig").DeviceId;

/// Mirrors `ggml_metal_device_t` (ggml-metal-device.h:14 @c1d0e7a00): an
/// opaque handle whose definition lives in the Objective-C.
pub const Device = opaque {};
/// Mirrors `ggml_metal_buffer_t` (ggml-metal-device.h:324 @c1d0e7a00).
pub const Buffer = opaque {};
/// Mirrors `ggml_metal_event_t` (ggml-metal-device.h:291 @c1d0e7a00).
pub const Event = opaque {};
/// Mirrors `ggml_metal_t` (ggml-metal-context.h:13 @c1d0e7a00).
pub const Context = opaque {};

/// Mirrors `ggml_metal_library_t` (ggml-metal-device.h:98 @c1d0e7a00).
pub const Library = opaque {};
/// Mirrors `ggml_metal_pipeline_t` (ggml-metal-device.h:33 @c1d0e7a00).
pub const Pipeline = opaque {};
/// Mirrors `ggml_metal_cv_t` (ggml-metal-device.h:20 @c1d0e7a00): a
/// wrapper over `MTLFunctionConstantValues`.
pub const Cv = opaque {};
/// Mirrors `ggml_metal_pipelines_t` (ggml-metal-device.h:39 @c1d0e7a00).
/// The struct behind it is ours — `library.zig` defines it, since
/// `ggml-metal-device.cpp` did.
pub const Pipelines = opaque {};

/// Mirrors `struct ggml_metal_pipeline_with_params` (ggml-metal-device.h:47
/// @c1d0e7a00).
///
/// **Returned by value by all 68 `get_pipeline_*` functions**, which makes
/// its ABI load-bearing: it is 40 bytes with interior padding, and Zig
/// 0.16 is known to mishandle small `extern struct`s *received* by value
/// (see `CLAUDE.md`). `make struct-layout` checks the layout and the
/// return path, because a silent mismatch here would corrupt every
/// pipeline lookup at once.
pub const PipelineWithParams = extern struct {
    pipeline: ?*Pipeline,

    nsg: c_int,

    nr0: c_int,
    nr1: c_int,

    smem: usize,

    c4: bool,
    cnt: bool,
};

/// Mirrors `struct ggml_metal_device_props` (ggml-metal-device.h:264
/// @c1d0e7a00).
///
/// Every field is present because the offsets have to match, not because
/// this port reads them. `make struct-layout` checks that they do.
pub const DeviceProps = extern struct {
    device: c_int,
    device_phys: c_int,
    device_virt: c_int,
    name: [128]u8,
    desc: [128]u8,

    max_buffer_size: usize,
    max_working_set_size: usize,
    max_theadgroup_memory_size: usize,

    has_simdgroup_reduction: bool,
    has_simdgroup_mm: bool,
    has_unified_memory: bool,
    has_bfloat: bool,
    has_tensor: bool,
    use_residency_sets: bool,
    use_shared_buffers: bool,

    supports_gpu_family_apple7: bool,

    device_id: DeviceId,
    gpu_family: c_int,

    op_offload_min_batch_size: c_int,
};

// -----------------------------------------------------------------------------
// ggml-metal-device.h

pub extern fn ggml_metal_device_id_token(id: DeviceId) [*c]const u8;
pub extern fn ggml_metal_device_get(device: c_int, n_devices: c_int) ?*Device;
pub extern fn ggml_metal_device_get_props(dev: *Device) *const DeviceProps;
pub extern fn ggml_metal_device_event_init(dev: *Device) ?*Event;
pub extern fn ggml_metal_device_event_free(dev: *Device, ev: *Event) void;
pub extern fn ggml_metal_device_event_synchronize(dev: *Device, ev: *Event) void;
pub extern fn ggml_metal_device_get_memory(dev: *Device, free: *usize, total: *usize) void;
pub extern fn ggml_metal_device_supports_op(dev: *Device, op: *const Tensor) bool;

pub extern fn ggml_metal_device_init(device: c_int, n_devices: c_int) ?*Device;
pub extern fn ggml_metal_device_free(dev: *Device) void;

pub extern fn ggml_metal_cv_init() ?*Cv;
pub extern fn ggml_metal_cv_free(cv: *Cv) void;
pub extern fn ggml_metal_cv_set_int16(cv: *Cv, value: i16, idx: i32) void;
pub extern fn ggml_metal_cv_set_int32(cv: *Cv, value: i32, idx: i32) void;
pub extern fn ggml_metal_cv_set_bool(cv: *Cv, value: bool, idx: i32) void;

/// `ggml_metal_pipeline_init` (ggml-metal-device.h:35 @c1d0e7a00).
///
/// Lives in the `.m`; it only `calloc`s the wrapper and leaves `obj` nil,
/// which is why the unit tests can make real handles without a device.
/// `struct ggml_metal_buffer_id` (ggml-metal-device.h:9 @c1d0e7a00).
///
/// A Metal buffer handle and a byte offset into it. 16 bytes, which is one
/// of the shapes the Zig 0.16 by-value-struct ABI bug does **not** affect
/// — but it is both returned by value from `ggml_metal_buffer_get_id` and
/// passed by value to `ggml_metal_encoder_set_buffer`, so
/// `make struct-layout` checks its layout and round-trips one rather than
/// relying on that measurement holding.
pub const BufferId = extern struct {
    /// `id<MTLBuffer>`.
    metal: ?*anyopaque,
    offs: usize,
};

/// `ggml_metal_cmd_buf_t` (ggml-metal-device.h:67 @c1d0e7a00) — an opaque
/// `id<MTLCommandBuffer>`, which the C spells as `void *`.
pub const CmdBuf = ?*anyopaque;

/// `ggml_metal_encoder_t` (ggml-metal-device.h:73 @c1d0e7a00).
pub const Encoder = opaque {};

// The encoder surface `ggml-metal-ops.cpp` drives. All of it lives in
// `ggml-metal-device.m`, which stays Objective-C (Decision 13).

pub extern fn ggml_metal_encoder_init(cmd_buf_raw: CmdBuf, concurrent: bool) ?*Encoder;
pub extern fn ggml_metal_encoder_free(encoder: *Encoder) void;
pub extern fn ggml_metal_encoder_debug_group_push(encoder: *Encoder, name: [*:0]const u8) void;
pub extern fn ggml_metal_encoder_debug_group_pop(encoder: *Encoder) void;
pub extern fn ggml_metal_encoder_set_pipeline(encoder: *Encoder, pipeline: PipelineWithParams) void;
pub extern fn ggml_metal_encoder_set_bytes(encoder: *Encoder, data: ?*anyopaque, size: usize, idx: c_int) void;
pub extern fn ggml_metal_encoder_set_buffer(encoder: *Encoder, buffer: BufferId, idx: c_int) void;
pub extern fn ggml_metal_encoder_set_threadgroup_memory_size(encoder: *Encoder, size: usize, idx: c_int) void;
pub extern fn ggml_metal_encoder_dispatch_threadgroups(encoder: *Encoder, tg0: c_int, tg1: c_int, tg2: c_int, tptg0: c_int, tptg1: c_int, tptg2: c_int) void;
pub extern fn ggml_metal_encoder_memory_barrier(encoder: *Encoder) void;
pub extern fn ggml_metal_encoder_end_encoding(encoder: *Encoder) void;

/// `ggml_metal_pipeline_max_theads_per_threadgroup`
/// (ggml-metal-device.h:61 @c1d0e7a00). The misspelling is upstream's.
pub extern fn ggml_metal_pipeline_max_theads_per_threadgroup(pipeline: PipelineWithParams) c_int;

/// `ggml_metal_device_get_library` (ggml-metal-device.h:304 @c1d0e7a00).
pub extern fn ggml_metal_device_get_library(dev: *Device) ?*Library;

/// `ggml_metal_buffer_get_id` (ggml-metal-device.h:343 @c1d0e7a00).
pub extern fn ggml_metal_buffer_get_id(buf: *Buffer, t: *const c.ggml_tensor) BufferId;

pub extern fn ggml_metal_pipeline_init() ?*Pipeline;
pub extern fn ggml_metal_pipeline_free(pipeline: *Pipeline) void;

pub extern fn ggml_metal_library_get_pipeline(lib: *Library, name: [*:0]const u8) PipelineWithParams;
pub extern fn ggml_metal_library_compile_pipeline(lib: *Library, base: [*:0]const u8, name: [*:0]const u8, cv: ?*Cv) PipelineWithParams;
pub extern fn ggml_metal_library_get_device(lib: *Library) *Device;

pub extern fn ggml_metal_buffer_init(dev: *Device, size: usize, shared: bool) ?*Buffer;
pub extern fn ggml_metal_buffer_map(dev: *Device, ptr: ?*anyopaque, size: usize, max_tensor_size: usize) ?*Buffer;
pub extern fn ggml_metal_buffer_free(buf: *Buffer) void;
pub extern fn ggml_metal_buffer_get_base(buf: *Buffer) ?*anyopaque;
pub extern fn ggml_metal_buffer_is_shared(buf: *Buffer) bool;
pub extern fn ggml_metal_buffer_memset_tensor(buf: *Buffer, tensor: *Tensor, value: u8, offset: usize, size: usize) void;
pub extern fn ggml_metal_buffer_set_tensor(buf: *Buffer, tensor: *Tensor, data: ?*const anyopaque, offset: usize, size: usize) void;
pub extern fn ggml_metal_buffer_get_tensor(buf: *Buffer, tensor: *const Tensor, data: ?*anyopaque, offset: usize, size: usize) void;
pub extern fn ggml_metal_buffer_cpy_tensor(buf: *Buffer, src: *const Tensor, dst: *Tensor) bool;
pub extern fn ggml_metal_buffer_clear(buf: *Buffer, value: u8) void;

// -----------------------------------------------------------------------------
// ggml-metal-context.h

pub extern fn ggml_metal_init(dev: *Device) ?*Context;
pub extern fn ggml_metal_free(ctx: *Context) void;
pub extern fn ggml_metal_get_name(ctx: *Context) [*c]const u8;
pub extern fn ggml_metal_synchronize(ctx: *Context) void;
pub extern fn ggml_metal_set_tensor_async(ctx: *Context, tensor: *Tensor, data: ?*const anyopaque, offset: usize, size: usize) void;
pub extern fn ggml_metal_get_tensor_async(ctx: *Context, tensor: *const Tensor, data: ?*anyopaque, offset: usize, size: usize) void;
pub extern fn ggml_metal_cpy_tensor_async(ctx_src: *Context, ctx_dst: *Context, src: *const Tensor, dst: *Tensor) bool;
pub extern fn ggml_metal_graph_compute(ctx: *Context, gf: *CGraph) c.enum_ggml_status;
pub extern fn ggml_metal_graph_optimize(ctx: *Context, gf: *CGraph) void;
pub extern fn ggml_metal_event_record(ctx: *Context, ev: *Event) void;
pub extern fn ggml_metal_event_wait(ctx: *Context, ev: *Event) void;
pub extern fn ggml_metal_set_n_cb(ctx: *Context, n_cb: c_int) void;
pub extern fn ggml_metal_set_abort_callback(ctx: *Context, abort_callback: c.ggml_abort_callback, user_data: ?*anyopaque) void;
pub extern fn ggml_metal_supports_family(ctx: *Context, family: c_int) bool;
pub extern fn ggml_metal_capture_next_compute(ctx: *Context) void;

// -----------------------------------------------------------------------------
// ggml-metal-ops.h

pub extern fn ggml_metal_op_mul_mat_id_extra_tpe(op: *const Tensor) usize;
pub extern fn ggml_metal_op_mul_mat_id_extra_ids(op: *const Tensor) usize;
pub extern fn ggml_metal_op_flash_attn_ext_extra_pad(op: *const Tensor) usize;
pub extern fn ggml_metal_op_flash_attn_ext_extra_blk(op: *const Tensor) usize;
pub extern fn ggml_metal_op_flash_attn_ext_extra_tmp(op: *const Tensor) usize;
pub extern fn ggml_metal_op_flash_attn_ext_extra_kv_f16(op: *const Tensor) usize;

// -----------------------------------------------------------------------------
// The layout gate's view of `DeviceProps`
//
// `harness/struct_layout.cpp` includes the real header and compares these
// against its own `sizeof` and `offsetof`. Exported rather than asserted
// at comptime because only the C compiler knows the answer.

export fn zz_props_sizeof() usize {
    return @sizeOf(DeviceProps);
}

/// The same facts for `PipelineWithParams`, plus a value the C side reads
/// back to prove the **return** ABI — 40 bytes with interior padding,
/// returned by value 68 times over.
export fn zz_pwp_sizeof() usize {
    return @sizeOf(PipelineWithParams);
}
export fn zz_pwp_nfields() usize {
    return @typeInfo(PipelineWithParams).@"struct".fields.len;
}
export fn zz_pwp_offset(i: usize) usize {
    const fields = @typeInfo(PipelineWithParams).@"struct".fields;
    inline for (fields, 0..) |f, k| {
        if (k == i) return @offsetOf(PipelineWithParams, f.name);
    }
    return std.math.maxInt(usize);
}
export fn zz_pwp_field_size(i: usize) usize {
    const fields = @typeInfo(PipelineWithParams).@"struct".fields;
    inline for (fields, 0..) |f, k| {
        if (k == i) return @sizeOf(f.type);
    }
    return std.math.maxInt(usize);
}

/// Returns a struct with every field set to a distinct marker, so the C
/// side can confirm each one survives the return.
export fn zz_pwp_roundtrip() PipelineWithParams {
    return .{
        .pipeline = @ptrFromInt(0xdead0000),
        .nsg = 11,
        .nr0 = 22,
        .nr1 = 33,
        .smem = 44444,
        .c4 = true,
        .cnt = false,
    };
}
export fn zz_props_alignof() usize {
    return @alignOf(DeviceProps);
}
export fn zz_props_nfields() usize {
    return @typeInfo(DeviceProps).@"struct".fields.len;
}

/// The offset of field `i`, in declaration order, or `maxInt` past the end.
///
/// One function over an index rather than 22 named exports, so adding a
/// field upstream cannot leave a new one unchecked: the harness compares
/// `zz_props_nfields` too.
export fn zz_props_offset(i: usize) usize {
    const fields = @typeInfo(DeviceProps).@"struct".fields;
    inline for (fields, 0..) |f, k| {
        if (k == i) return @offsetOf(DeviceProps, f.name);
    }
    return std.math.maxInt(usize);
}

/// The size of field `i`, which catches a field of the right offset and
/// the wrong width — `c_int` where the C has `size_t`, say.
export fn zz_props_field_size(i: usize) usize {
    const fields = @typeInfo(DeviceProps).@"struct".fields;
    inline for (fields, 0..) |f, k| {
        if (k == i) return @sizeOf(f.type);
    }
    return std.math.maxInt(usize);
}

// `BufferId`'s layout and round-trip, for `harness/struct_layout.cpp`.

export fn zz_bid_sizeof() usize {
    return @sizeOf(BufferId);
}

export fn zz_bid_alignof() usize {
    return @alignOf(BufferId);
}

export fn zz_bid_nfields() usize {
    return @typeInfo(BufferId).@"struct".fields.len;
}

export fn zz_bid_offset(i: usize) usize {
    inline for (@typeInfo(BufferId).@"struct".fields, 0..) |f, k| {
        if (k == i) return @offsetOf(BufferId, f.name);
    }
    return std.math.maxInt(usize);
}

export fn zz_bid_field_size(i: usize) usize {
    inline for (@typeInfo(BufferId).@"struct".fields, 0..) |f, k| {
        if (k == i) return @sizeOf(f.type);
    }
    return std.math.maxInt(usize);
}

/// Takes a `BufferId` **by value** from C and hands the two fields back,
/// so the gate can see whether the receive path carries them. This is the
/// direction Zig 0.16 gets wrong for some shapes.
export fn zz_bid_roundtrip(in: BufferId, out_metal: *?*anyopaque, out_offs: *usize) void {
    out_metal.* = in.metal;
    out_offs.* = in.offs;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the props struct has the field count the gate expects" {
    // `make struct-layout` compares this against the C's; asserting it here
    // too means `zig build test` notices a field added on our side alone.
    try std.testing.expectEqual(@as(usize, 19), zz_props_nfields());
}

test "the device id enum matches the table's" {
    // `DeviceProps.device_id` is typed as the same enum the tuning table
    // indexes, so a renumbering there would silently change what this
    // port reads out of the props.
    try std.testing.expectEqual(@as(c_uint, 15), @intFromEnum(DeviceId.m4_max));
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(DeviceId));
}
