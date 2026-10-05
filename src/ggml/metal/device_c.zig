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
