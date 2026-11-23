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
//! all at v1.0.0 (`b1b770d90`).
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
//! Hand-transcription is the largest source of silent error here, and `DeviceProps` below is 18 fields of which this port
//! reads a few — the rest exist only to place those at the right
//! offsets. Getting one wrong reads an adjacent field and reports
//! nothing.
//!
//! **`make struct-layout` is the gate.** It compiles the real header and
//! compares `sizeof` and every `offsetof` against the values this file
//! exports. Negative-tested by perturbing a field.

const std = @import("std");
const ggml = @import("../module.zig");

/// Mirrors `enum ggml_metal_device_id` (ggml-metal-device.h:268 @b1b770d90).
///
/// Written by `testing/scripts/gen-tuning-table`, in the enum's own order: the props
/// field below has its width, and the port only carries it through to
/// `ggml_metal_device_id_token`. Do not edit the members; rerun the script.
pub const DeviceId = enum(c_uint) {
    generic = 0,
    m1,
    m1_pro,
    m1_max,
    m1_ultra,
    m2,
    m2_pro,
    m2_max,
    m2_ultra,
    m3,
    m3_pro,
    m3_max,
    m3_ultra,
    m4,
    m4_pro,
    m4_max,
    m5,
    m5_pro,
    m5_max,
    m5_ultra,
    a18_pro,
};

/// Mirrors `ggml_metal_device_t` (ggml-metal-device.h:14 @b1b770d90): an
/// opaque handle whose definition lives in the Objective-C.
pub const Device = opaque {};
/// Mirrors `ggml_metal_buffer_t` (ggml-metal-device.h:358 @b1b770d90).
pub const Buffer = opaque {};
/// Mirrors `ggml_metal_event_t` (ggml-metal-device.h:320 @b1b770d90).
pub const Event = opaque {};
/// Mirrors `ggml_metal_t` (ggml-metal-context.h:13 @b1b770d90).
pub const Context = opaque {};

/// Mirrors `ggml_metal_library_t` (ggml-metal-device.h:98 @b1b770d90).
pub const Library = opaque {};
/// Mirrors `ggml_metal_pipeline_t` (ggml-metal-device.h:33 @b1b770d90).
pub const Pipeline = opaque {};
/// Mirrors `ggml_metal_cv_t` (ggml-metal-device.h:20 @b1b770d90): a
/// wrapper over `MTLFunctionConstantValues`.
pub const Cv = opaque {};
/// Mirrors `ggml_metal_pipelines_t` (ggml-metal-device.h:39 @b1b770d90).
/// The struct behind it is ours — `library.zig` defines it, since
/// `ggml-metal-device.cpp` did.
pub const Pipelines = opaque {};

/// Mirrors `struct ggml_metal_pipeline_with_params` (ggml-metal-device.h:47
/// @b1b770d90).
///
/// **Returned by value by all 76 `get_pipeline_*` functions**, which makes
/// its ABI load-bearing: it is 40 bytes with interior padding, and Zig
/// 0.16 is known to mishandle small `extern struct`s *received* by value.
/// `make struct-layout` checks the layout and the
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

/// Mirrors `struct ggml_metal_device_props` (ggml-metal-device.h:295
/// @b1b770d90). The run of `bool`s ends at `use_shared_buffers`;
/// the byte after it is padding before `device_id`.
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

    device_id: DeviceId,
    gpu_family: c_int,

    op_offload_min_batch_size: c_int,
};

// -----------------------------------------------------------------------------
// ggml-metal-device.h

pub extern fn ggml_metal_device_id_token(id: DeviceId) [*c]const u8;
pub extern fn ggml_metal_device_get_props(dev: *Device) *const DeviceProps;
/// `ggml_metal_device_get_fusion_info` (ggml-metal-device.h:352
/// @b1b770d90): the device-owned fusion info, defined in `fusion.zig` and
/// created by `ggml-metal-device.m`.
pub extern fn ggml_metal_device_get_fusion_info(dev: *Device) ?*ggml.metal.fusion.Info;
pub extern fn ggml_metal_device_event_init(dev: *Device) ?*Event;
pub extern fn ggml_metal_device_event_free(dev: *Device, ev: *Event) void;
pub extern fn ggml_metal_device_event_synchronize(dev: *Device, ev: *Event) void;
pub extern fn ggml_metal_device_get_memory(dev: *Device, free: *usize, total: *usize) void;
pub extern fn ggml_metal_device_supports_op(dev: *Device, op: *const ggml.impl.ggml_c.ggml_tensor) bool;

pub extern fn ggml_metal_device_init(device: c_int, n_devices: c_int) ?*Device;
pub extern fn ggml_metal_device_free(dev: *Device) void;

pub extern fn ggml_metal_cv_init() ?*Cv;
pub extern fn ggml_metal_cv_free(cv: *Cv) void;
pub extern fn ggml_metal_cv_set_int16(cv: *Cv, value: i16, idx: i32) void;
pub extern fn ggml_metal_cv_set_int32(cv: *Cv, value: i32, idx: i32) void;
pub extern fn ggml_metal_cv_set_bool(cv: *Cv, value: bool, idx: i32) void;

/// `ggml_metal_pipeline_init` (ggml-metal-device.h:35 @b1b770d90).
///
/// Lives in the `.m`; it only `calloc`s the wrapper and leaves `obj` nil,
/// which is why the unit tests can make real handles without a device.
/// `struct ggml_metal_buffer_id` (ggml-metal-device.h:9 @b1b770d90).
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

/// `ggml_metal_cmd_buf_t` (ggml-metal-device.h:67 @b1b770d90) — an opaque
/// `id<MTLCommandBuffer>`, which the C spells as `void *`.
pub const CmdBuf = ?*anyopaque;

/// `ggml_metal_encoder_t` (ggml-metal-device.h:73 @b1b770d90).
pub const Encoder = opaque {};

// The encoder surface `ggml-metal-ops.cpp` drives. All of it lives in
// `ggml-metal-device.m`, which stays Objective-C.

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
/// (ggml-metal-device.h:61 @b1b770d90). The misspelling is upstream's.
pub extern fn ggml_metal_pipeline_max_theads_per_threadgroup(pipeline: PipelineWithParams) c_int;

/// `ggml_metal_device_get_library` (ggml-metal-device.h:333 @b1b770d90).
pub extern fn ggml_metal_device_get_library(dev: *Device) ?*Library;

/// `ggml_metal_buffer_get_id` (ggml-metal-device.h:380 @b1b770d90).
pub extern fn ggml_metal_buffer_get_id(buf: *Buffer, t: *const ggml.impl.ggml_c.ggml_tensor) BufferId;

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
pub extern fn ggml_metal_buffer_memset_tensor(buf: *Buffer, tensor: *ggml.impl.ggml_c.ggml_tensor, value: u8, offset: usize, size: usize) void;
pub extern fn ggml_metal_buffer_set_tensor(buf: *Buffer, tensor: *ggml.impl.ggml_c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) void;
pub extern fn ggml_metal_buffer_get_tensor(buf: *Buffer, tensor: *const ggml.impl.ggml_c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) void;
pub extern fn ggml_metal_buffer_set_tensor_n(buf: *Buffer, tensors: [*c]const [*c]ggml.impl.ggml_c.ggml_tensor, data: [*c]const ?*const anyopaque, offsets: [*c]const usize, sizes: [*c]const usize, n_tensors: usize) void;
pub extern fn ggml_metal_buffer_get_tensor_n(buf: *Buffer, tensors: [*c]const [*c]const ggml.impl.ggml_c.ggml_tensor, data: [*c]const ?*anyopaque, offsets: [*c]const usize, sizes: [*c]const usize, n_tensors: usize) void;
pub extern fn ggml_metal_buffer_cpy_tensor(buf: *Buffer, src: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) bool;
pub extern fn ggml_metal_buffer_cpy_tensor_n(buf: *Buffer, srcs: [*c]const [*c]const ggml.impl.ggml_c.ggml_tensor, dsts: [*c]const [*c]ggml.impl.ggml_c.ggml_tensor, n_tensors: usize) bool;
pub extern fn ggml_metal_buffer_clear(buf: *Buffer, value: u8) void;

// -----------------------------------------------------------------------------
// ggml-metal-context.h

pub extern fn ggml_metal_init(dev: *Device) ?*Context;
pub extern fn ggml_metal_free(ctx: *Context) void;
pub extern fn ggml_metal_get_name(ctx: *Context) [*c]const u8;
pub extern fn ggml_metal_synchronize(ctx: *Context) void;
pub extern fn ggml_metal_set_tensor_async(ctx: *Context, tensor: *ggml.impl.ggml_c.ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) void;
pub extern fn ggml_metal_get_tensor_async(ctx: *Context, tensor: *const ggml.impl.ggml_c.ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) void;
pub extern fn ggml_metal_cpy_tensor_async(ctx_src: *Context, ctx_dst: *Context, src: *const ggml.impl.ggml_c.ggml_tensor, dst: *ggml.impl.ggml_c.ggml_tensor) bool;
pub extern fn ggml_metal_graph_compute(ctx: *Context, gf: *ggml.impl.ggml_c.ggml_cgraph) ggml.impl.ggml_c.enum_ggml_status;
pub extern fn ggml_metal_graph_optimize(ctx: *Context, gf: *ggml.impl.ggml_c.ggml_cgraph) void;
pub extern fn ggml_metal_event_record(ctx: *Context, ev: *Event) void;
pub extern fn ggml_metal_event_wait(ctx: *Context, ev: *Event) void;
pub extern fn ggml_metal_set_n_cb(ctx: *Context, n_cb: c_int) void;
pub extern fn ggml_metal_set_abort_callback(ctx: *Context, abort_callback: ggml.impl.ggml_c.ggml_abort_callback, user_data: ?*anyopaque) void;
pub extern fn ggml_metal_supports_family(ctx: *Context, family: c_int) bool;
pub extern fn ggml_metal_capture_next_compute(ctx: *Context) void;

// -----------------------------------------------------------------------------
// ggml-metal-ops.h
//
// The scratch-size queries are the port's (`ops.zig`), as is
// `ggml_metal_device_get` (`library.zig`): callers reach them through the
// barrel; they stay exported for upstream's `ggml-metal.cpp` in `metal-diff`.

// -----------------------------------------------------------------------------
// The layout gate's view of `DeviceProps`
//
// `testing/harness/struct_layout.cpp` includes the real header and compares these
// against its own `sizeof` and `offsetof`. Exported rather than asserted
// at comptime because only the C compiler knows the answer.

pub fn zz_props_sizeof() callconv(.c) usize {
    return @sizeOf(DeviceProps);
}

/// The same facts for `PipelineWithParams`, plus a value the C side reads
/// back to prove the **return** ABI — 40 bytes with interior padding,
/// returned by value 76 times over.
pub fn zz_pwp_sizeof() callconv(.c) usize {
    return @sizeOf(PipelineWithParams);
}
pub fn zz_pwp_nfields() callconv(.c) usize {
    return @typeInfo(PipelineWithParams).@"struct".field_names.len;
}
pub fn zz_pwp_offset(i: usize) callconv(.c) usize {
    inline for (@typeInfo(PipelineWithParams).@"struct".field_names, 0..) |name, k| {
        if (k == i) return @offsetOf(PipelineWithParams, name);
    }
    return std.math.maxInt(usize);
}
pub fn zz_pwp_field_size(i: usize) callconv(.c) usize {
    inline for (@typeInfo(PipelineWithParams).@"struct".field_types, 0..) |F, k| {
        if (k == i) return @sizeOf(F);
    }
    return std.math.maxInt(usize);
}

/// Returns a struct with every field set to a distinct marker, so the C
/// side can confirm each one survives the return.
pub fn zz_pwp_roundtrip() callconv(.c) PipelineWithParams {
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
pub fn zz_props_alignof() callconv(.c) usize {
    return @alignOf(DeviceProps);
}
pub fn zz_props_nfields() callconv(.c) usize {
    return @typeInfo(DeviceProps).@"struct".field_names.len;
}

/// The offset of field `i`, in declaration order, or `maxInt` past the end.
///
/// One function over an index rather than 22 named exports, so adding a
/// field upstream cannot leave a new one unchecked: the harness compares
/// `zz_props_nfields` too.
pub fn zz_props_offset(i: usize) callconv(.c) usize {
    inline for (@typeInfo(DeviceProps).@"struct".field_names, 0..) |name, k| {
        if (k == i) return @offsetOf(DeviceProps, name);
    }
    return std.math.maxInt(usize);
}

/// The size of field `i`, which catches a field of the right offset and
/// the wrong width — `c_int` where the C has `size_t`, say.
pub fn zz_props_field_size(i: usize) callconv(.c) usize {
    inline for (@typeInfo(DeviceProps).@"struct".field_types, 0..) |F, k| {
        if (k == i) return @sizeOf(F);
    }
    return std.math.maxInt(usize);
}

// `BufferId`'s layout and round-trip, for `testing/harness/struct_layout.cpp`.

pub fn zz_bid_sizeof() callconv(.c) usize {
    return @sizeOf(BufferId);
}

pub fn zz_bid_alignof() callconv(.c) usize {
    return @alignOf(BufferId);
}

pub fn zz_bid_nfields() callconv(.c) usize {
    return @typeInfo(BufferId).@"struct".field_names.len;
}

pub fn zz_bid_offset(i: usize) callconv(.c) usize {
    inline for (@typeInfo(BufferId).@"struct".field_names, 0..) |name, k| {
        if (k == i) return @offsetOf(BufferId, name);
    }
    return std.math.maxInt(usize);
}

pub fn zz_bid_field_size(i: usize) callconv(.c) usize {
    inline for (@typeInfo(BufferId).@"struct".field_types, 0..) |F, k| {
        if (k == i) return @sizeOf(F);
    }
    return std.math.maxInt(usize);
}

/// Takes a `BufferId` **by value** from C and hands the two fields back,
/// so the gate can see whether the receive path carries them. This is the
/// direction Zig 0.16 gets wrong for some shapes.
pub fn zz_bid_roundtrip(in: BufferId, out_metal: *?*anyopaque, out_offs: *usize) callconv(.c) void {
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
    try std.testing.expectEqual(@as(usize, 18), zz_props_nfields());
}

test "the device id enum has the C's width and order" {
    // `DeviceProps.device_id` is typed as this enum, so its width places
    // every later field, and `ggml_metal_device_id_token` reads its value.
    try std.testing.expectEqual(@as(c_uint, 15), @backingInt(DeviceId.m4_max));
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(DeviceId));
}
