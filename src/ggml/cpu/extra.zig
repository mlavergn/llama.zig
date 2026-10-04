//! The extra-buffer accelerator interface: `ggml::cpu::tensor_traits` and
//! `ggml::cpu::extra_buffer_type`, and the two C entry points that walk it.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml-cpu/traits.cpp` — the two entry points
//! - `llama.cpp/ggml/src/ggml-cpu/traits.h`   — the two abstract classes
//!
//! both at v0.3.0 (`c1d0e7a00`).
//!
//! # Why this file forces a four-way port
//!
//! These two classes are the only C++ coupling in `ggml-cpu/`. Measured, the
//! whole of it is five symbols: `~tensor_traits`, `~extra_buffer_type`, the
//! typeinfo for both, and `ggml_backend_cpu_get_extra_buffer_types()`. The
//! first four are referenced by `repack.cpp`, which derives from the bases;
//! the fifth by `traits.cpp`, which calls it. So `traits.cpp`, `repack.cpp`,
//! `arch/arm/repack.cpp` and `ggml-cpu.cpp` move together or not at all.
//!
//! Everything else crossing those four files is a plain C symbol.
//!
//! # The vtable is explicit
//!
//! C++ puts a hidden vptr at offset 0 and emits the table itself. Zig has no
//! inheritance, so the table is a `VTable` struct of function pointers and
//! the "base class" is a struct whose first field points at one. Derived
//! types embed that struct as their first field, so a `*Derived` converts to
//! the base by pointer cast — the same layout C++ chose, written out.
//!
//! No destructor entry: the two C++ dtors exist only because the bases are
//! polymorphic, and nothing here is ever deleted through a base pointer.
//! The instances are `static` or leaked `new`s that live for the process.

const std = @import("std");
const impl = @import("../impl.zig");
const defs = @import("defs.zig");

const c = impl.c;

pub const Tensor = defs.Tensor;
pub const ComputeParams = defs.ComputeParams;

/// Ports `tensor_traits` (ggml-cpu/traits.h:20 @c1d0e7a00).
///
/// Registered in `tensor->extra` by whatever buffer type owns the weights.
pub const TensorTraits = extern struct {
    vtable: *const VTable,

    pub const VTable = extern struct {
        /// `virtual bool work_size(int, const ggml_tensor *, size_t &)`.
        ///
        /// The C++ takes `size_t &`; a reference is a pointer in the ABI, so
        /// it is spelled as one here.
        work_size: *const fn (self: *const TensorTraits, n_threads: c_int, op: *const Tensor, size: *usize) callconv(.c) bool,

        /// `virtual bool compute_forward(ggml_compute_params *, ggml_tensor *)`.
        compute_forward: *const fn (self: *const TensorTraits, params: *ComputeParams, op: *Tensor) callconv(.c) bool,
    };
};

/// Ports `extra_buffer_type` (ggml-cpu/traits.h:27
/// @c1d0e7a00), the context hanging off an extra `ggml_backend_buffer_type`.
pub const ExtraBufferType = extern struct {
    vtable: *const VTable,

    pub const VTable = extern struct {
        /// `virtual bool supports_op(ggml_backend_dev_t, const ggml_tensor *)`.
        supports_op: *const fn (self: *const ExtraBufferType, dev: c.ggml_backend_dev_t, op: *const Tensor) callconv(.c) bool,

        /// `virtual tensor_traits * get_tensor_traits(const ggml_tensor *)`.
        get_tensor_traits: *const fn (self: *const ExtraBufferType, op: *const Tensor) callconv(.c) ?*const TensorTraits,
    };
};

/// The list `ggml-cpu.cpp` builds. Declared here rather than imported so the
/// two files keep the C's direction of dependency: `traits.cpp` calls into
/// `ggml-cpu.cpp`, not the other way round.
///
/// The C returns `std::vector<ggml_backend_buffer_type_t> &`; the port
/// returns a slice over a static array, which is what that vector is once
/// its one-time initialiser has run.
const cpu_backend = @import("cpu_backend.zig");

/// Ports `ggml_cpu_extra_compute_forward` (traits.cpp:12 @c1d0e7a00).
///
/// Parameters:
/// - `params`: this thread's slice of the work.
/// - `op`: the node to compute.
///
/// Return: true when an accelerator claimed the op and has computed it, in
/// which case the caller skips its own kernel.
pub export fn ggml_cpu_extra_compute_forward(params: *ComputeParams, op: *Tensor) callconv(.c) bool {
    for (cpu_backend.getExtraBufferTypes()) |extra| {
        const buft = extra orelse continue;
        const ctx = buft.*.context orelse continue;
        const buf_extra: *const ExtraBufferType = @ptrCast(@alignCast(ctx));
        if (buf_extra.vtable.get_tensor_traits(buf_extra, op)) |traits| {
            if (traits.vtable.compute_forward(traits, params, op)) return true;
        }
    }
    return false;
}

/// Ports `ggml_cpu_extra_work_size` (traits.cpp:25 @c1d0e7a00).
///
/// Parameters:
/// - `n_threads`: the thread count the plan is being sized for.
/// - `op`: the node.
/// - `size`: receives the work-buffer bytes the accelerator needs.
///
/// Return: true when an accelerator claimed the op and set `size`.
pub export fn ggml_cpu_extra_work_size(n_threads: c_int, op: *const Tensor, size: *usize) callconv(.c) bool {
    for (cpu_backend.getExtraBufferTypes()) |extra| {
        const buft = extra orelse continue;
        const ctx = buft.*.context orelse continue;
        const buf_extra: *const ExtraBufferType = @ptrCast(@alignCast(ctx));
        if (buf_extra.vtable.get_tensor_traits(buf_extra, op)) |traits| {
            if (traits.vtable.work_size(traits, n_threads, op, size)) return true;
        }
    }
    return false;
}
