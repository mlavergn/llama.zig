//! libggml's C API: every symbol the archive exports, by its Zig path.
//!
//! # Provenance
//!
//! **Not a port.** The ported files declare plain `pub fn ... callconv(.c)`;
//! this root is the one place they become C symbols. `module.zig` forces it
//! when libggml itself is the compilation root, and under test. A consumer
//! that imports the ggml module and calls it as Zig gets only what it uses;
//! a gate that compiles one file against upstream's C gets none of these.
//! Backend and architecture symbols export under the condition their files
//! compile in.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config");
const mod = @import("module.zig");

comptime {
    @setEvalBranchQuota(1_000_000);
    // alloc.zig
    @export(&mod.alloc.ggml_op_can_inplace, .{ .name = "ggml_op_can_inplace" });
    @export(&mod.alloc.ggml_tallocr_new, .{ .name = "ggml_tallocr_new" });
    @export(&mod.alloc.ggml_tallocr_alloc, .{ .name = "ggml_tallocr_alloc" });
    @export(&mod.alloc.ggml_gallocr_new_n, .{ .name = "ggml_gallocr_new_n" });
    @export(&mod.alloc.ggml_gallocr_new, .{ .name = "ggml_gallocr_new" });
    @export(&mod.alloc.ggml_gallocr_free, .{ .name = "ggml_gallocr_free" });
    @export(&mod.alloc.ggml_gallocr_reserve_n_size, .{ .name = "ggml_gallocr_reserve_n_size" });
    @export(&mod.alloc.ggml_gallocr_reserve_n, .{ .name = "ggml_gallocr_reserve_n" });
    @export(&mod.alloc.ggml_gallocr_reserve, .{ .name = "ggml_gallocr_reserve" });
    @export(&mod.alloc.ggml_gallocr_alloc_graph, .{ .name = "ggml_gallocr_alloc_graph" });
    @export(&mod.alloc.ggml_gallocr_get_buffer_size, .{ .name = "ggml_gallocr_get_buffer_size" });
    @export(&mod.alloc.ggml_backend_alloc_ctx_tensors_from_buft, .{ .name = "ggml_backend_alloc_ctx_tensors_from_buft" });
    @export(&mod.alloc.ggml_backend_alloc_ctx_tensors_from_buft_size, .{ .name = "ggml_backend_alloc_ctx_tensors_from_buft_size" });
    @export(&mod.alloc.ggml_backend_alloc_ctx_tensors, .{ .name = "ggml_backend_alloc_ctx_tensors" });
    // backend.zig
    @export(&mod.backend.ggml_backend_buft_name, .{ .name = "ggml_backend_buft_name" });
    @export(&mod.backend.ggml_backend_buft_alloc_buffer, .{ .name = "ggml_backend_buft_alloc_buffer" });
    @export(&mod.backend.ggml_backend_buft_alloc_buffer_n, .{ .name = "ggml_backend_buft_alloc_buffer_n" });
    @export(&mod.backend.ggml_backend_buft_get_alignment, .{ .name = "ggml_backend_buft_get_alignment" });
    @export(&mod.backend.ggml_backend_buft_get_max_size, .{ .name = "ggml_backend_buft_get_max_size" });
    @export(&mod.backend.ggml_backend_buft_get_alloc_size, .{ .name = "ggml_backend_buft_get_alloc_size" });
    @export(&mod.backend.ggml_backend_buft_get_alloc_size_n, .{ .name = "ggml_backend_buft_get_alloc_size_n" });
    @export(&mod.backend.ggml_backend_buft_is_host, .{ .name = "ggml_backend_buft_is_host" });
    @export(&mod.backend.ggml_backend_buft_get_device, .{ .name = "ggml_backend_buft_get_device" });
    @export(&mod.backend.ggml_backend_buffer_init, .{ .name = "ggml_backend_buffer_init" });
    @export(&mod.backend.ggml_backend_buffer_name, .{ .name = "ggml_backend_buffer_name" });
    @export(&mod.backend.ggml_backend_buffer_free, .{ .name = "ggml_backend_buffer_free" });
    @export(&mod.backend.ggml_backend_buffer_get_size, .{ .name = "ggml_backend_buffer_get_size" });
    @export(&mod.backend.ggml_backend_buffer_get_base, .{ .name = "ggml_backend_buffer_get_base" });
    @export(&mod.backend.ggml_backend_buffer_init_tensor, .{ .name = "ggml_backend_buffer_init_tensor" });
    @export(&mod.backend.ggml_backend_buffer_clear, .{ .name = "ggml_backend_buffer_clear" });
    @export(&mod.backend.ggml_backend_buffer_get_alignment, .{ .name = "ggml_backend_buffer_get_alignment" });
    @export(&mod.backend.ggml_backend_buffer_get_max_size, .{ .name = "ggml_backend_buffer_get_max_size" });
    @export(&mod.backend.ggml_backend_buffer_get_alloc_size, .{ .name = "ggml_backend_buffer_get_alloc_size" });
    @export(&mod.backend.ggml_backend_buffer_is_host, .{ .name = "ggml_backend_buffer_is_host" });
    @export(&mod.backend.ggml_backend_buffer_set_usage, .{ .name = "ggml_backend_buffer_set_usage" });
    @export(&mod.backend.ggml_backend_buffer_get_usage, .{ .name = "ggml_backend_buffer_get_usage" });
    @export(&mod.backend.ggml_backend_buffer_get_type, .{ .name = "ggml_backend_buffer_get_type" });
    @export(&mod.backend.ggml_backend_buffer_reset, .{ .name = "ggml_backend_buffer_reset" });
    @export(&mod.backend.ggml_backend_buffer_copy_tensor, .{ .name = "ggml_backend_buffer_copy_tensor" });
    @export(&mod.backend.ggml_backend_guid, .{ .name = "ggml_backend_guid" });
    @export(&mod.backend.ggml_backend_name, .{ .name = "ggml_backend_name" });
    @export(&mod.backend.ggml_backend_free, .{ .name = "ggml_backend_free" });
    @export(&mod.backend.ggml_backend_get_default_buffer_type, .{ .name = "ggml_backend_get_default_buffer_type" });
    @export(&mod.backend.ggml_backend_alloc_buffer, .{ .name = "ggml_backend_alloc_buffer" });
    @export(&mod.backend.ggml_backend_get_alignment, .{ .name = "ggml_backend_get_alignment" });
    @export(&mod.backend.ggml_backend_get_max_size, .{ .name = "ggml_backend_get_max_size" });
    @export(&mod.backend.ggml_backend_tensor_set_async, .{ .name = "ggml_backend_tensor_set_async" });
    @export(&mod.backend.ggml_backend_tensor_get_async, .{ .name = "ggml_backend_tensor_get_async" });
    @export(&mod.backend.ggml_backend_tensor_set_2d_async, .{ .name = "ggml_backend_tensor_set_2d_async" });
    @export(&mod.backend.ggml_backend_tensor_get_2d_async, .{ .name = "ggml_backend_tensor_get_2d_async" });
    @export(&mod.backend.ggml_backend_tensor_set, .{ .name = "ggml_backend_tensor_set" });
    @export(&mod.backend.ggml_backend_tensor_get, .{ .name = "ggml_backend_tensor_get" });
    @export(&mod.backend.ggml_backend_tensor_set_2d, .{ .name = "ggml_backend_tensor_set_2d" });
    @export(&mod.backend.ggml_backend_tensor_get_2d, .{ .name = "ggml_backend_tensor_get_2d" });
    @export(&mod.backend.ggml_backend_tensor_set_n, .{ .name = "ggml_backend_tensor_set_n" });
    @export(&mod.backend.ggml_backend_tensor_get_n, .{ .name = "ggml_backend_tensor_get_n" });
    @export(&mod.backend.ggml_backend_tensor_memset, .{ .name = "ggml_backend_tensor_memset" });
    @export(&mod.backend.ggml_backend_synchronize, .{ .name = "ggml_backend_synchronize" });
    @export(&mod.backend.ggml_backend_graph_plan_create, .{ .name = "ggml_backend_graph_plan_create" });
    @export(&mod.backend.ggml_backend_graph_plan_free, .{ .name = "ggml_backend_graph_plan_free" });
    @export(&mod.backend.ggml_backend_graph_plan_compute, .{ .name = "ggml_backend_graph_plan_compute" });
    @export(&mod.backend.ggml_backend_graph_compute, .{ .name = "ggml_backend_graph_compute" });
    @export(&mod.backend.ggml_backend_graph_compute_async, .{ .name = "ggml_backend_graph_compute_async" });
    @export(&mod.backend.ggml_backend_supports_op, .{ .name = "ggml_backend_supports_op" });
    @export(&mod.backend.ggml_backend_supports_buft, .{ .name = "ggml_backend_supports_buft" });
    @export(&mod.backend.ggml_backend_offload_op, .{ .name = "ggml_backend_offload_op" });
    @export(&mod.backend.ggml_backend_get_device, .{ .name = "ggml_backend_get_device" });
    @export(&mod.backend.ggml_backend_tensor_copy, .{ .name = "ggml_backend_tensor_copy" });
    @export(&mod.backend.ggml_backend_tensor_copy_n, .{ .name = "ggml_backend_tensor_copy_n" });
    @export(&mod.backend.ggml_backend_tensor_copy_async, .{ .name = "ggml_backend_tensor_copy_async" });
    @export(&mod.backend.ggml_backend_event_new, .{ .name = "ggml_backend_event_new" });
    @export(&mod.backend.ggml_backend_event_free, .{ .name = "ggml_backend_event_free" });
    @export(&mod.backend.ggml_backend_event_record, .{ .name = "ggml_backend_event_record" });
    @export(&mod.backend.ggml_backend_event_synchronize, .{ .name = "ggml_backend_event_synchronize" });
    @export(&mod.backend.ggml_backend_event_wait, .{ .name = "ggml_backend_event_wait" });
    @export(&mod.backend.ggml_backend_dev_name, .{ .name = "ggml_backend_dev_name" });
    @export(&mod.backend.ggml_backend_dev_description, .{ .name = "ggml_backend_dev_description" });
    @export(&mod.backend.ggml_backend_dev_memory, .{ .name = "ggml_backend_dev_memory" });
    @export(&mod.backend.ggml_backend_dev_type, .{ .name = "ggml_backend_dev_type" });
    @export(&mod.backend.ggml_backend_dev_get_props, .{ .name = "ggml_backend_dev_get_props" });
    @export(&mod.backend.ggml_backend_dev_backend_reg, .{ .name = "ggml_backend_dev_backend_reg" });
    @export(&mod.backend.ggml_backend_dev_init, .{ .name = "ggml_backend_dev_init" });
    @export(&mod.backend.ggml_backend_dev_buffer_type, .{ .name = "ggml_backend_dev_buffer_type" });
    @export(&mod.backend.ggml_backend_dev_host_buffer_type, .{ .name = "ggml_backend_dev_host_buffer_type" });
    @export(&mod.backend.ggml_backend_dev_buffer_from_host_ptr, .{ .name = "ggml_backend_dev_buffer_from_host_ptr" });
    @export(&mod.backend.ggml_backend_dev_supports_op, .{ .name = "ggml_backend_dev_supports_op" });
    @export(&mod.backend.ggml_backend_dev_supports_buft, .{ .name = "ggml_backend_dev_supports_buft" });
    @export(&mod.backend.ggml_backend_dev_offload_op, .{ .name = "ggml_backend_dev_offload_op" });
    @export(&mod.backend.ggml_backend_reg_name, .{ .name = "ggml_backend_reg_name" });
    @export(&mod.backend.ggml_backend_reg_dev_count, .{ .name = "ggml_backend_reg_dev_count" });
    @export(&mod.backend.ggml_backend_reg_dev_get, .{ .name = "ggml_backend_reg_dev_get" });
    @export(&mod.backend.ggml_backend_reg_get_proc_address, .{ .name = "ggml_backend_reg_get_proc_address" });
    @export(&mod.backend.ggml_backend_multi_buffer_alloc_buffer, .{ .name = "ggml_backend_multi_buffer_alloc_buffer" });
    @export(&mod.backend.ggml_backend_buffer_is_multi_buffer, .{ .name = "ggml_backend_buffer_is_multi_buffer" });
    @export(&mod.backend.ggml_backend_multi_buffer_set_usage, .{ .name = "ggml_backend_multi_buffer_set_usage" });
    @export(&mod.backend.ggml_op_alloc_size_may_expand, .{ .name = "ggml_op_alloc_size_may_expand" });
    @export(&mod.backend.ggml_backend_view_init, .{ .name = "ggml_backend_view_init" });
    @export(&mod.backend.ggml_backend_tensor_alloc, .{ .name = "ggml_backend_tensor_alloc" });
    @export(&mod.backend.ggml_backend_graph_copy, .{ .name = "ggml_backend_graph_copy" });
    @export(&mod.backend.ggml_backend_graph_copy_free, .{ .name = "ggml_backend_graph_copy_free" });
    @export(&mod.backend.ggml_backend_compare_graph_backend, .{ .name = "ggml_backend_compare_graph_backend" });
    @export(&mod.backend.ggml_backend_cpu_buffer_type, .{ .name = "ggml_backend_cpu_buffer_type" });
    @export(&mod.backend.ggml_backend_cpu_buffer_from_ptr, .{ .name = "ggml_backend_cpu_buffer_from_ptr" });
    // backend_meta.zig
    @export(&mod.backend_meta.ggml_backend_meta_split_axis_name, .{ .name = "ggml_backend_meta_split_axis_name" });
    @export(&mod.backend_meta.ggml_backend_meta_device, .{ .name = "ggml_backend_meta_device" });
    @export(&mod.backend_meta.ggml_backend_buft_is_meta, .{ .name = "ggml_backend_buft_is_meta" });
    @export(&mod.backend_meta.ggml_backend_buffer_is_meta, .{ .name = "ggml_backend_buffer_is_meta" });
    @export(&mod.backend_meta.ggml_backend_meta_buffer_set_usage, .{ .name = "ggml_backend_meta_buffer_set_usage" });
    @export(&mod.backend_meta.ggml_backend_is_meta, .{ .name = "ggml_backend_is_meta" });
    @export(&mod.backend_meta.ggml_backend_meta_n_backends, .{ .name = "ggml_backend_meta_n_backends" });
    @export(&mod.backend_meta.ggml_backend_meta_simple_backend, .{ .name = "ggml_backend_meta_simple_backend" });
    // backend_reg.zig
    @export(&mod.backend_reg.ggml_backend_register, .{ .name = "ggml_backend_register" });
    @export(&mod.backend_reg.ggml_backend_device_register, .{ .name = "ggml_backend_device_register" });
    @export(&mod.backend_reg.ggml_backend_reg_count, .{ .name = "ggml_backend_reg_count" });
    @export(&mod.backend_reg.ggml_backend_reg_get, .{ .name = "ggml_backend_reg_get" });
    @export(&mod.backend_reg.ggml_backend_reg_by_name, .{ .name = "ggml_backend_reg_by_name" });
    @export(&mod.backend_reg.ggml_backend_dev_count, .{ .name = "ggml_backend_dev_count" });
    @export(&mod.backend_reg.ggml_backend_dev_get, .{ .name = "ggml_backend_dev_get" });
    @export(&mod.backend_reg.ggml_backend_dev_by_name, .{ .name = "ggml_backend_dev_by_name" });
    @export(&mod.backend_reg.ggml_backend_dev_by_type, .{ .name = "ggml_backend_dev_by_type" });
    @export(&mod.backend_reg.ggml_backend_init_by_name, .{ .name = "ggml_backend_init_by_name" });
    @export(&mod.backend_reg.ggml_backend_init_by_type, .{ .name = "ggml_backend_init_by_type" });
    @export(&mod.backend_reg.ggml_backend_init_best, .{ .name = "ggml_backend_init_best" });
    @export(&mod.backend_reg.ggml_backend_load, .{ .name = "ggml_backend_load" });
    @export(&mod.backend_reg.ggml_backend_unload, .{ .name = "ggml_backend_unload" });
    @export(&mod.backend_reg.ggml_backend_load_all, .{ .name = "ggml_backend_load_all" });
    @export(&mod.backend_reg.ggml_backend_load_all_from_path, .{ .name = "ggml_backend_load_all_from_path" });
    // backend_sched.zig
    @export(&mod.backend_sched.ggml_backend_sched_split_graph, .{ .name = "ggml_backend_sched_split_graph" });
    @export(&mod.backend_sched.ggml_backend_sched_new, .{ .name = "ggml_backend_sched_new" });
    @export(&mod.backend_sched.ggml_backend_sched_free, .{ .name = "ggml_backend_sched_free" });
    @export(&mod.backend_sched.ggml_backend_sched_reset, .{ .name = "ggml_backend_sched_reset" });
    @export(&mod.backend_sched.ggml_backend_sched_reserve_size, .{ .name = "ggml_backend_sched_reserve_size" });
    @export(&mod.backend_sched.ggml_backend_sched_reserve, .{ .name = "ggml_backend_sched_reserve" });
    @export(&mod.backend_sched.ggml_backend_sched_alloc_graph, .{ .name = "ggml_backend_sched_alloc_graph" });
    @export(&mod.backend_sched.ggml_backend_sched_graph_compute, .{ .name = "ggml_backend_sched_graph_compute" });
    @export(&mod.backend_sched.ggml_backend_sched_graph_compute_async, .{ .name = "ggml_backend_sched_graph_compute_async" });
    @export(&mod.backend_sched.ggml_backend_sched_synchronize, .{ .name = "ggml_backend_sched_synchronize" });
    @export(&mod.backend_sched.ggml_backend_sched_set_eval_callback, .{ .name = "ggml_backend_sched_set_eval_callback" });
    @export(&mod.backend_sched.ggml_backend_sched_set_copy_callback, .{ .name = "ggml_backend_sched_set_copy_callback" });
    @export(&mod.backend_sched.ggml_backend_sched_set_realloc_callback, .{ .name = "ggml_backend_sched_set_realloc_callback" });
    @export(&mod.backend_sched.ggml_backend_sched_get_n_splits, .{ .name = "ggml_backend_sched_get_n_splits" });
    @export(&mod.backend_sched.ggml_backend_sched_get_n_copies, .{ .name = "ggml_backend_sched_get_n_copies" });
    @export(&mod.backend_sched.ggml_backend_sched_get_n_backends, .{ .name = "ggml_backend_sched_get_n_backends" });
    @export(&mod.backend_sched.ggml_backend_sched_get_backend, .{ .name = "ggml_backend_sched_get_backend" });
    @export(&mod.backend_sched.ggml_backend_sched_get_buffer_type, .{ .name = "ggml_backend_sched_get_buffer_type" });
    @export(&mod.backend_sched.ggml_backend_sched_get_buffer_size, .{ .name = "ggml_backend_sched_get_buffer_size" });
    @export(&mod.backend_sched.ggml_backend_sched_set_tensor_backend, .{ .name = "ggml_backend_sched_set_tensor_backend" });
    @export(&mod.backend_sched.ggml_backend_sched_get_tensor_backend, .{ .name = "ggml_backend_sched_get_tensor_backend" });
    // context.zig
    @export(&mod.context.ggml_init, .{ .name = "ggml_init" });
    @export(&mod.context.ggml_reset, .{ .name = "ggml_reset" });
    @export(&mod.context.ggml_free, .{ .name = "ggml_free" });
    @export(&mod.context.ggml_used_mem, .{ .name = "ggml_used_mem" });
    @export(&mod.context.ggml_get_no_alloc, .{ .name = "ggml_get_no_alloc" });
    @export(&mod.context.ggml_set_no_alloc, .{ .name = "ggml_set_no_alloc" });
    @export(&mod.context.ggml_get_mem_buffer, .{ .name = "ggml_get_mem_buffer" });
    @export(&mod.context.ggml_get_mem_size, .{ .name = "ggml_get_mem_size" });
    @export(&mod.context.ggml_get_max_tensor_size, .{ .name = "ggml_get_max_tensor_size" });
    @export(&mod.context.ggml_new_buffer, .{ .name = "ggml_new_buffer" });
    @export(&mod.context.ggml_new_tensor, .{ .name = "ggml_new_tensor" });
    @export(&mod.context.ggml_new_tensor_1d, .{ .name = "ggml_new_tensor_1d" });
    @export(&mod.context.ggml_new_tensor_2d, .{ .name = "ggml_new_tensor_2d" });
    @export(&mod.context.ggml_new_tensor_3d, .{ .name = "ggml_new_tensor_3d" });
    @export(&mod.context.ggml_new_tensor_4d, .{ .name = "ggml_new_tensor_4d" });
    @export(&mod.context.ggml_dup_tensor, .{ .name = "ggml_dup_tensor" });
    @export(&mod.context.ggml_view_tensor, .{ .name = "ggml_view_tensor" });
    @export(&mod.context.ggml_get_first_tensor, .{ .name = "ggml_get_first_tensor" });
    @export(&mod.context.ggml_get_next_tensor, .{ .name = "ggml_get_next_tensor" });
    @export(&mod.context.ggml_get_tensor, .{ .name = "ggml_get_tensor" });
    @export(&mod.context.ggml_unravel_index, .{ .name = "ggml_unravel_index" });
    @export(&mod.context.ggml_get_data, .{ .name = "ggml_get_data" });
    @export(&mod.context.ggml_get_data_f32, .{ .name = "ggml_get_data_f32" });
    @export(&mod.context.ggml_get_name, .{ .name = "ggml_get_name" });
    @export(&mod.context.ggml_set_name, .{ .name = "ggml_set_name" });
    @export(&mod.context.ggml_print_object, .{ .name = "ggml_print_object" });
    @export(&mod.context.ggml_print_objects, .{ .name = "ggml_print_objects" });
    // cpu/binary_ops.zig
    @export(&mod.cpu.binary_ops.ggml_compute_forward_add_non_quantized, .{ .name = "ggml_compute_forward_add_non_quantized" });
    @export(&mod.cpu.binary_ops.ggml_compute_forward_sub, .{ .name = "ggml_compute_forward_sub" });
    @export(&mod.cpu.binary_ops.ggml_compute_forward_mul, .{ .name = "ggml_compute_forward_mul" });
    @export(&mod.cpu.binary_ops.ggml_compute_forward_div, .{ .name = "ggml_compute_forward_div" });
    // cpu/convert.zig
    @export(&mod.cpu.convert.ggml_cpu_fp32_to_fp32, .{ .name = "ggml_cpu_fp32_to_fp32" });
    @export(&mod.cpu.convert.ggml_cpu_fp32_to_fp16, .{ .name = "ggml_cpu_fp32_to_fp16" });
    @export(&mod.cpu.convert.ggml_cpu_fp16_to_fp32, .{ .name = "ggml_cpu_fp16_to_fp32" });
    @export(&mod.cpu.convert.ggml_cpu_fp32_to_bf16, .{ .name = "ggml_cpu_fp32_to_bf16" });
    @export(&mod.cpu.convert.ggml_cpu_bf16_to_fp32, .{ .name = "ggml_cpu_bf16_to_fp32" });
    @export(&mod.cpu.convert.ggml_cpu_fp32_to_i32, .{ .name = "ggml_cpu_fp32_to_i32" });
    // cpu/cpu_backend.zig
    @export(&mod.cpu.cpu_backend.ggml_backend_cpu_init, .{ .name = "ggml_backend_cpu_init" });
    @export(&mod.cpu.cpu_backend.ggml_backend_is_cpu, .{ .name = "ggml_backend_is_cpu" });
    @export(&mod.cpu.cpu_backend.ggml_backend_cpu_set_n_threads, .{ .name = "ggml_backend_cpu_set_n_threads" });
    @export(&mod.cpu.cpu_backend.ggml_backend_cpu_set_threadpool, .{ .name = "ggml_backend_cpu_set_threadpool" });
    @export(&mod.cpu.cpu_backend.ggml_backend_cpu_set_abort_callback, .{ .name = "ggml_backend_cpu_set_abort_callback" });
    @export(&mod.cpu.cpu_backend.ggml_backend_cpu_set_use_ref, .{ .name = "ggml_backend_cpu_set_use_ref" });
    @export(&mod.cpu.cpu_backend.ggml_backend_cpu_reg, .{ .name = "ggml_backend_cpu_reg" });
    // cpu/extra.zig
    @export(&mod.cpu.extra.ggml_cpu_extra_compute_forward, .{ .name = "ggml_cpu_extra_compute_forward" });
    @export(&mod.cpu.extra.ggml_cpu_extra_work_size, .{ .name = "ggml_cpu_extra_work_size" });
    // cpu/features.zig
    @export(&mod.cpu.features.ggml_cpu_has_avx, .{ .name = "ggml_cpu_has_avx" });
    @export(&mod.cpu.features.ggml_cpu_has_avx_vnni, .{ .name = "ggml_cpu_has_avx_vnni" });
    @export(&mod.cpu.features.ggml_cpu_has_avx2, .{ .name = "ggml_cpu_has_avx2" });
    @export(&mod.cpu.features.ggml_cpu_has_avx512, .{ .name = "ggml_cpu_has_avx512" });
    @export(&mod.cpu.features.ggml_cpu_has_avx512_vbmi, .{ .name = "ggml_cpu_has_avx512_vbmi" });
    @export(&mod.cpu.features.ggml_cpu_has_avx512_vnni, .{ .name = "ggml_cpu_has_avx512_vnni" });
    @export(&mod.cpu.features.ggml_cpu_has_avx512_bf16, .{ .name = "ggml_cpu_has_avx512_bf16" });
    @export(&mod.cpu.features.ggml_cpu_has_amx_int8, .{ .name = "ggml_cpu_has_amx_int8" });
    @export(&mod.cpu.features.ggml_cpu_has_bmi2, .{ .name = "ggml_cpu_has_bmi2" });
    @export(&mod.cpu.features.ggml_cpu_has_fma, .{ .name = "ggml_cpu_has_fma" });
    @export(&mod.cpu.features.ggml_cpu_has_f16c, .{ .name = "ggml_cpu_has_f16c" });
    @export(&mod.cpu.features.ggml_cpu_has_sse3, .{ .name = "ggml_cpu_has_sse3" });
    @export(&mod.cpu.features.ggml_cpu_has_ssse3, .{ .name = "ggml_cpu_has_ssse3" });
    @export(&mod.cpu.features.ggml_cpu_has_vsx, .{ .name = "ggml_cpu_has_vsx" });
    @export(&mod.cpu.features.ggml_cpu_has_vxe, .{ .name = "ggml_cpu_has_vxe" });
    @export(&mod.cpu.features.ggml_cpu_has_wasm_simd, .{ .name = "ggml_cpu_has_wasm_simd" });
    @export(&mod.cpu.features.ggml_cpu_has_riscv_v, .{ .name = "ggml_cpu_has_riscv_v" });
    @export(&mod.cpu.features.ggml_cpu_get_rvv_vlen, .{ .name = "ggml_cpu_get_rvv_vlen" });
    @export(&mod.cpu.features.ggml_cpu_has_arm_fma, .{ .name = "ggml_cpu_has_arm_fma" });
    @export(&mod.cpu.features.ggml_cpu_has_fp16_va, .{ .name = "ggml_cpu_has_fp16_va" });
    @export(&mod.cpu.features.ggml_cpu_has_neon, .{ .name = "ggml_cpu_has_neon" });
    @export(&mod.cpu.features.ggml_cpu_has_dotprod, .{ .name = "ggml_cpu_has_dotprod" });
    @export(&mod.cpu.features.ggml_cpu_has_sve, .{ .name = "ggml_cpu_has_sve" });
    @export(&mod.cpu.features.ggml_cpu_has_matmul_int8, .{ .name = "ggml_cpu_has_matmul_int8" });
    @export(&mod.cpu.features.ggml_cpu_get_sve_cnt, .{ .name = "ggml_cpu_get_sve_cnt" });
    @export(&mod.cpu.features.ggml_cpu_has_sme, .{ .name = "ggml_cpu_has_sme" });
    @export(&mod.cpu.features.ggml_cpu_has_sme2, .{ .name = "ggml_cpu_has_sme2" });
    @export(&mod.cpu.features.ggml_cpu_has_llamafile, .{ .name = "ggml_cpu_has_llamafile" });
    @export(&mod.cpu.features.ggml_cpu_init, .{ .name = "ggml_cpu_init" });
    // cpu/mulmat.zig
    @export(&mod.cpu.mulmat.ggml_compute_forward_mul_mat, .{ .name = "ggml_compute_forward_mul_mat" });
    // cpu/ops/activation.zig
    @export(&mod.cpu.ops.activation.ggml_compute_forward_leaky_relu, .{ .name = "ggml_compute_forward_leaky_relu" });
    @export(&mod.cpu.ops.activation.ggml_compute_forward_silu_back, .{ .name = "ggml_compute_forward_silu_back" });
    @export(&mod.cpu.ops.activation.ggml_compute_forward_unary, .{ .name = "ggml_compute_forward_unary" });
    @export(&mod.cpu.ops.activation.ggml_compute_forward_glu, .{ .name = "ggml_compute_forward_glu" });
    // cpu/ops/arith.zig
    @export(&mod.cpu.ops.arith.ggml_compute_forward_add, .{ .name = "ggml_compute_forward_add" });
    @export(&mod.cpu.ops.arith.ggml_compute_forward_add_id, .{ .name = "ggml_compute_forward_add_id" });
    @export(&mod.cpu.ops.arith.ggml_compute_forward_add1, .{ .name = "ggml_compute_forward_add1" });
    @export(&mod.cpu.ops.arith.ggml_compute_forward_acc, .{ .name = "ggml_compute_forward_acc" });
    // cpu/ops/attention.zig
    @export(&mod.cpu.ops.attention.ggml_compute_forward_flash_attn_ext, .{ .name = "ggml_compute_forward_flash_attn_ext" });
    @export(&mod.cpu.ops.attention.ggml_compute_forward_flash_attn_back, .{ .name = "ggml_compute_forward_flash_attn_back" });
    @export(&mod.cpu.ops.attention.ggml_compute_forward_lightning_indexer, .{ .name = "ggml_compute_forward_lightning_indexer" });
    // cpu/ops/conv.zig
    @export(&mod.cpu.ops.conv.ggml_compute_forward_conv_transpose_1d, .{ .name = "ggml_compute_forward_conv_transpose_1d" });
    @export(&mod.cpu.ops.conv.ggml_compute_forward_im2col, .{ .name = "ggml_compute_forward_im2col" });
    @export(&mod.cpu.ops.conv.ggml_compute_forward_im2col_back_f32, .{ .name = "ggml_compute_forward_im2col_back_f32" });
    @export(&mod.cpu.ops.conv.ggml_compute_forward_im2col_3d, .{ .name = "ggml_compute_forward_im2col_3d" });
    @export(&mod.cpu.ops.conv.ggml_compute_forward_col2im_1d, .{ .name = "ggml_compute_forward_col2im_1d" });
    @export(&mod.cpu.ops.conv.ggml_compute_forward_conv_2d, .{ .name = "ggml_compute_forward_conv_2d" });
    @export(&mod.cpu.ops.conv.ggml_compute_forward_conv_3d, .{ .name = "ggml_compute_forward_conv_3d" });
    @export(&mod.cpu.ops.conv.ggml_compute_forward_conv_transpose_2d, .{ .name = "ggml_compute_forward_conv_transpose_2d" });
    @export(&mod.cpu.ops.conv.ggml_compute_forward_conv_2d_dw, .{ .name = "ggml_compute_forward_conv_2d_dw" });
    // cpu/ops/custom.zig
    @export(&mod.cpu.ops.custom.ggml_compute_forward_map_custom1, .{ .name = "ggml_compute_forward_map_custom1" });
    @export(&mod.cpu.ops.custom.ggml_compute_forward_map_custom2, .{ .name = "ggml_compute_forward_map_custom2" });
    @export(&mod.cpu.ops.custom.ggml_compute_forward_map_custom3, .{ .name = "ggml_compute_forward_map_custom3" });
    @export(&mod.cpu.ops.custom.ggml_compute_forward_custom, .{ .name = "ggml_compute_forward_custom" });
    @export(&mod.cpu.ops.custom.ggml_compute_forward_cross_entropy_loss, .{ .name = "ggml_compute_forward_cross_entropy_loss" });
    @export(&mod.cpu.ops.custom.ggml_compute_forward_cross_entropy_loss_back, .{ .name = "ggml_compute_forward_cross_entropy_loss_back" });
    @export(&mod.cpu.ops.custom.ggml_compute_forward_opt_step_adamw, .{ .name = "ggml_compute_forward_opt_step_adamw" });
    @export(&mod.cpu.ops.custom.ggml_compute_forward_opt_step_sgd, .{ .name = "ggml_compute_forward_opt_step_sgd" });
    @export(&mod.cpu.ops.custom.ggml_compute_forward_solve_tri, .{ .name = "ggml_compute_forward_solve_tri" });
    // cpu/ops/dup.zig
    @export(&mod.cpu.ops.dup.ggml_compute_forward_dup, .{ .name = "ggml_compute_forward_dup" });
    @export(&mod.cpu.ops.dup.ggml_compute_forward_cpy, .{ .name = "ggml_compute_forward_cpy" });
    @export(&mod.cpu.ops.dup.ggml_compute_forward_cont, .{ .name = "ggml_compute_forward_cont" });
    // cpu/ops/gather.zig
    @export(&mod.cpu.ops.gather.ggml_compute_forward_get_rows, .{ .name = "ggml_compute_forward_get_rows" });
    @export(&mod.cpu.ops.gather.ggml_compute_forward_set_rows, .{ .name = "ggml_compute_forward_set_rows" });
    @export(&mod.cpu.ops.gather.ggml_compute_forward_get_rows_back, .{ .name = "ggml_compute_forward_get_rows_back" });
    @export(&mod.cpu.ops.gather.ggml_compute_forward_diag, .{ .name = "ggml_compute_forward_diag" });
    // cpu/ops/linalg.zig
    @export(&mod.cpu.ops.linalg.ggml_compute_forward_out_prod, .{ .name = "ggml_compute_forward_out_prod" });
    @export(&mod.cpu.ops.linalg.ggml_compute_forward_scale, .{ .name = "ggml_compute_forward_scale" });
    @export(&mod.cpu.ops.linalg.ggml_compute_forward_set, .{ .name = "ggml_compute_forward_set" });
    // cpu/ops/norm.zig
    @export(&mod.cpu.ops.norm.ggml_compute_forward_norm, .{ .name = "ggml_compute_forward_norm" });
    @export(&mod.cpu.ops.norm.ggml_compute_forward_rms_norm, .{ .name = "ggml_compute_forward_rms_norm" });
    @export(&mod.cpu.ops.norm.ggml_compute_forward_rms_norm_mul_fused, .{ .name = "ggml_compute_forward_rms_norm_mul_fused" });
    @export(&mod.cpu.ops.norm.ggml_compute_forward_rms_norm_back, .{ .name = "ggml_compute_forward_rms_norm_back" });
    @export(&mod.cpu.ops.norm.ggml_compute_forward_group_norm, .{ .name = "ggml_compute_forward_group_norm" });
    @export(&mod.cpu.ops.norm.ggml_compute_forward_l2_norm, .{ .name = "ggml_compute_forward_l2_norm" });
    // cpu/ops/pad.zig
    @export(&mod.cpu.ops.pad.ggml_compute_forward_upscale, .{ .name = "ggml_compute_forward_upscale" });
    @export(&mod.cpu.ops.pad.ggml_compute_forward_pad, .{ .name = "ggml_compute_forward_pad" });
    @export(&mod.cpu.ops.pad.ggml_compute_forward_pad_reflect_1d, .{ .name = "ggml_compute_forward_pad_reflect_1d" });
    @export(&mod.cpu.ops.pad.ggml_compute_forward_roll, .{ .name = "ggml_compute_forward_roll" });
    @export(&mod.cpu.ops.pad.ggml_compute_forward_arange, .{ .name = "ggml_compute_forward_arange" });
    @export(&mod.cpu.ops.pad.ggml_compute_forward_timestep_embedding, .{ .name = "ggml_compute_forward_timestep_embedding" });
    // cpu/ops/pool.zig
    @export(&mod.cpu.ops.pool.ggml_compute_forward_pool_1d, .{ .name = "ggml_compute_forward_pool_1d" });
    @export(&mod.cpu.ops.pool.ggml_compute_forward_pool_2d, .{ .name = "ggml_compute_forward_pool_2d" });
    @export(&mod.cpu.ops.pool.ggml_compute_forward_pool_2d_back, .{ .name = "ggml_compute_forward_pool_2d_back" });
    // cpu/ops/recurrent.zig
    @export(&mod.cpu.ops.recurrent.ggml_compute_forward_rwkv_wkv6, .{ .name = "ggml_compute_forward_rwkv_wkv6" });
    @export(&mod.cpu.ops.recurrent.ggml_compute_forward_gla, .{ .name = "ggml_compute_forward_gla" });
    @export(&mod.cpu.ops.recurrent.ggml_compute_forward_gated_delta_net, .{ .name = "ggml_compute_forward_gated_delta_net" });
    @export(&mod.cpu.ops.recurrent.ggml_compute_forward_dsv4_hc_comb, .{ .name = "ggml_compute_forward_dsv4_hc_comb" });
    @export(&mod.cpu.ops.recurrent.ggml_compute_forward_dsv4_hc_pre, .{ .name = "ggml_compute_forward_dsv4_hc_pre" });
    @export(&mod.cpu.ops.recurrent.ggml_compute_forward_dsv4_hc_post, .{ .name = "ggml_compute_forward_dsv4_hc_post" });
    @export(&mod.cpu.ops.recurrent.ggml_compute_forward_rwkv_wkv7, .{ .name = "ggml_compute_forward_rwkv_wkv7" });
    // cpu/ops/reduce.zig
    @export(&mod.cpu.ops.reduce.ggml_compute_forward_sum, .{ .name = "ggml_compute_forward_sum" });
    @export(&mod.cpu.ops.reduce.ggml_compute_forward_cumsum, .{ .name = "ggml_compute_forward_cumsum" });
    @export(&mod.cpu.ops.reduce.ggml_compute_forward_sum_rows, .{ .name = "ggml_compute_forward_sum_rows" });
    @export(&mod.cpu.ops.reduce.ggml_compute_forward_mean, .{ .name = "ggml_compute_forward_mean" });
    @export(&mod.cpu.ops.reduce.ggml_compute_forward_argmax, .{ .name = "ggml_compute_forward_argmax" });
    @export(&mod.cpu.ops.reduce.ggml_compute_forward_count_equal, .{ .name = "ggml_compute_forward_count_equal" });
    // cpu/ops/repeat.zig
    @export(&mod.cpu.ops.repeat.ggml_compute_forward_repeat, .{ .name = "ggml_compute_forward_repeat" });
    @export(&mod.cpu.ops.repeat.ggml_compute_forward_repeat_back, .{ .name = "ggml_compute_forward_repeat_back" });
    @export(&mod.cpu.ops.repeat.ggml_compute_forward_concat, .{ .name = "ggml_compute_forward_concat" });
    // cpu/ops/rope.zig
    @export(&mod.cpu.ops.rope.ggml_compute_forward_rope, .{ .name = "ggml_compute_forward_rope" });
    @export(&mod.cpu.ops.rope.ggml_compute_forward_rope_back, .{ .name = "ggml_compute_forward_rope_back" });
    // cpu/ops/sgemm.zig
    @export(&mod.cpu.ops.sgemm.llamafile_sgemm, .{ .name = "llamafile_sgemm" });
    // cpu/ops/softmax.zig
    @export(&mod.cpu.ops.softmax.ggml_compute_forward_soft_max, .{ .name = "ggml_compute_forward_soft_max" });
    @export(&mod.cpu.ops.softmax.ggml_compute_forward_soft_max_ext_back, .{ .name = "ggml_compute_forward_soft_max_ext_back" });
    @export(&mod.cpu.ops.softmax.ggml_compute_forward_clamp, .{ .name = "ggml_compute_forward_clamp" });
    // cpu/ops/sort.zig
    @export(&mod.cpu.ops.sort.ggml_compute_forward_fill, .{ .name = "ggml_compute_forward_fill" });
    @export(&mod.cpu.ops.sort.ggml_compute_forward_tri, .{ .name = "ggml_compute_forward_tri" });
    @export(&mod.cpu.ops.sort.ggml_compute_forward_argsort, .{ .name = "ggml_compute_forward_argsort" });
    @export(&mod.cpu.ops.sort.ggml_compute_forward_top_k, .{ .name = "ggml_compute_forward_top_k" });
    @export(&mod.cpu.ops.sort.ggml_compute_forward_fwht, .{ .name = "ggml_compute_forward_fwht" });
    @export(&mod.cpu.ops.sort.ggml_compute_forward_top_k_mask, .{ .name = "ggml_compute_forward_top_k_mask" });
    // cpu/ops/ssm.zig
    @export(&mod.cpu.ops.ssm.ggml_compute_forward_ssm_conv, .{ .name = "ggml_compute_forward_ssm_conv" });
    @export(&mod.cpu.ops.ssm.ggml_compute_forward_ssm_scan, .{ .name = "ggml_compute_forward_ssm_scan" });
    // cpu/ops/window.zig
    @export(&mod.cpu.ops.window.ggml_compute_forward_win_part, .{ .name = "ggml_compute_forward_win_part" });
    @export(&mod.cpu.ops.window.ggml_compute_forward_win_unpart, .{ .name = "ggml_compute_forward_win_unpart" });
    @export(&mod.cpu.ops.window.ggml_compute_forward_get_rel_pos, .{ .name = "ggml_compute_forward_get_rel_pos" });
    @export(&mod.cpu.ops.window.ggml_compute_forward_add_rel_pos, .{ .name = "ggml_compute_forward_add_rel_pos" });
    // cpu/plan.zig
    @export(&mod.cpu.plan.ggml_graph_plan, .{ .name = "ggml_graph_plan" });
    // cpu/quants/iq.zig
    @export(&mod.cpu.quants.iq.ggml_vec_dot_iq2_xxs_q8_K_generic, .{ .name = "ggml_vec_dot_iq2_xxs_q8_K_generic" });
    @export(&mod.cpu.quants.iq.ggml_vec_dot_iq2_xs_q8_K_generic, .{ .name = "ggml_vec_dot_iq2_xs_q8_K_generic" });
    @export(&mod.cpu.quants.iq.ggml_vec_dot_iq2_s_q8_K_generic, .{ .name = "ggml_vec_dot_iq2_s_q8_K_generic" });
    @export(&mod.cpu.quants.iq.ggml_vec_dot_iq3_xxs_q8_K_generic, .{ .name = "ggml_vec_dot_iq3_xxs_q8_K_generic" });
    @export(&mod.cpu.quants.iq.ggml_vec_dot_iq3_s_q8_K_generic, .{ .name = "ggml_vec_dot_iq3_s_q8_K_generic" });
    @export(&mod.cpu.quants.iq.ggml_vec_dot_iq1_s_q8_K_generic, .{ .name = "ggml_vec_dot_iq1_s_q8_K_generic" });
    @export(&mod.cpu.quants.iq.ggml_vec_dot_iq1_m_q8_K_generic, .{ .name = "ggml_vec_dot_iq1_m_q8_K_generic" });
    @export(&mod.cpu.quants.iq.ggml_vec_dot_iq4_xs_q8_K_generic, .{ .name = "ggml_vec_dot_iq4_xs_q8_K_generic" });
    // cpu/quants/k.zig
    @export(&mod.cpu.quants.k.ggml_vec_dot_q2_K_q8_K_generic, .{ .name = "ggml_vec_dot_q2_K_q8_K_generic" });
    @export(&mod.cpu.quants.k.ggml_vec_dot_q3_K_q8_K_generic, .{ .name = "ggml_vec_dot_q3_K_q8_K_generic" });
    @export(&mod.cpu.quants.k.ggml_vec_dot_q4_K_q8_K_generic, .{ .name = "ggml_vec_dot_q4_K_q8_K_generic" });
    @export(&mod.cpu.quants.k.ggml_vec_dot_q5_K_q8_K_generic, .{ .name = "ggml_vec_dot_q5_K_q8_K_generic" });
    @export(&mod.cpu.quants.k.ggml_vec_dot_q6_K_q8_K_generic, .{ .name = "ggml_vec_dot_q6_K_q8_K_generic" });
    // cpu/quants/legacy.zig
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_q1_0_q8_0_generic, .{ .name = "ggml_vec_dot_q1_0_q8_0_generic" });
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_q2_0_q8_0_generic, .{ .name = "ggml_vec_dot_q2_0_q8_0_generic" });
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_q4_0_q8_0_generic, .{ .name = "ggml_vec_dot_q4_0_q8_0_generic" });
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_q4_1_q8_1_generic, .{ .name = "ggml_vec_dot_q4_1_q8_1_generic" });
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_mxfp4_q8_0_generic, .{ .name = "ggml_vec_dot_mxfp4_q8_0_generic" });
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_nvfp4_q8_0_generic, .{ .name = "ggml_vec_dot_nvfp4_q8_0_generic" });
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_f8_e4m3_f32, .{ .name = "ggml_vec_dot_f8_e4m3_f32" });
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_q5_0_q8_0_generic, .{ .name = "ggml_vec_dot_q5_0_q8_0_generic" });
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_q5_1_q8_1_generic, .{ .name = "ggml_vec_dot_q5_1_q8_1_generic" });
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_q8_0_q8_0_generic, .{ .name = "ggml_vec_dot_q8_0_q8_0_generic" });
    @export(&mod.cpu.quants.legacy.ggml_vec_dot_iq4_nl_q8_0_generic, .{ .name = "ggml_vec_dot_iq4_nl_q8_0_generic" });
    // cpu/quants/rows.zig
    @export(&mod.cpu.quants.rows.quantize_row_q1_0, .{ .name = "quantize_row_q1_0" });
    @export(&mod.cpu.quants.rows.quantize_row_q2_0, .{ .name = "quantize_row_q2_0" });
    @export(&mod.cpu.quants.rows.quantize_row_q4_0, .{ .name = "quantize_row_q4_0" });
    @export(&mod.cpu.quants.rows.quantize_row_q4_1, .{ .name = "quantize_row_q4_1" });
    @export(&mod.cpu.quants.rows.quantize_row_q5_0, .{ .name = "quantize_row_q5_0" });
    @export(&mod.cpu.quants.rows.quantize_row_q5_1, .{ .name = "quantize_row_q5_1" });
    @export(&mod.cpu.quants.rows.quantize_row_q8_0_generic, .{ .name = "quantize_row_q8_0_generic" });
    @export(&mod.cpu.quants.rows.quantize_row_q8_1_generic, .{ .name = "quantize_row_q8_1_generic" });
    @export(&mod.cpu.quants.rows.quantize_row_mxfp4, .{ .name = "quantize_row_mxfp4" });
    @export(&mod.cpu.quants.rows.quantize_row_nvfp4, .{ .name = "quantize_row_nvfp4" });
    @export(&mod.cpu.quants.rows.quantize_row_f8_e4m3, .{ .name = "quantize_row_f8_e4m3" });
    @export(&mod.cpu.quants.rows.quantize_row_q2_K, .{ .name = "quantize_row_q2_K" });
    @export(&mod.cpu.quants.rows.quantize_row_q3_K, .{ .name = "quantize_row_q3_K" });
    @export(&mod.cpu.quants.rows.quantize_row_q4_K, .{ .name = "quantize_row_q4_K" });
    @export(&mod.cpu.quants.rows.quantize_row_q5_K, .{ .name = "quantize_row_q5_K" });
    @export(&mod.cpu.quants.rows.quantize_row_q6_K, .{ .name = "quantize_row_q6_K" });
    @export(&mod.cpu.quants.rows.quantize_row_tq1_0, .{ .name = "quantize_row_tq1_0" });
    @export(&mod.cpu.quants.rows.quantize_row_tq2_0, .{ .name = "quantize_row_tq2_0" });
    @export(&mod.cpu.quants.rows.quantize_row_q8_K_generic, .{ .name = "quantize_row_q8_K_generic" });
    @export(&mod.cpu.quants.rows.quantize_row_iq4_nl, .{ .name = "quantize_row_iq4_nl" });
    @export(&mod.cpu.quants.rows.quantize_row_iq4_xs, .{ .name = "quantize_row_iq4_xs" });
    // cpu/quants/ternary.zig
    @export(&mod.cpu.quants.ternary.ggml_vec_dot_tq1_0_q8_K_generic, .{ .name = "ggml_vec_dot_tq1_0_q8_K_generic" });
    @export(&mod.cpu.quants.ternary.ggml_vec_dot_tq2_0_q8_K_generic, .{ .name = "ggml_vec_dot_tq2_0_q8_K_generic" });
    // cpu/repack/nibble4.zig
    @export(&mod.cpu.repack.nibble4.ggml_gemv_iq4_nl_4x4_q8_0_generic, .{ .name = "ggml_gemv_iq4_nl_4x4_q8_0_generic" });
    @export(&mod.cpu.repack.nibble4.ggml_gemm_iq4_nl_4x4_q8_0_generic, .{ .name = "ggml_gemm_iq4_nl_4x4_q8_0_generic" });
    @export(&mod.cpu.repack.nibble4.ggml_gemv_mxfp4_4x4_q8_0_generic, .{ .name = "ggml_gemv_mxfp4_4x4_q8_0_generic" });
    @export(&mod.cpu.repack.nibble4.ggml_gemm_mxfp4_4x4_q8_0_generic, .{ .name = "ggml_gemm_mxfp4_4x4_q8_0_generic" });
    // cpu/repack/q1_0.zig
    @export(&mod.cpu.repack.q1_0.ggml_gemv_q1_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q1_0_4x4_q8_0_generic" });
    @export(&mod.cpu.repack.q1_0.ggml_gemv_q1_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q1_0_4x8_q8_0_generic" });
    @export(&mod.cpu.repack.q1_0.ggml_gemm_q1_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q1_0_4x4_q8_0_generic" });
    @export(&mod.cpu.repack.q1_0.ggml_gemm_q1_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q1_0_4x8_q8_0_generic" });
    // cpu/repack/q4_0.zig
    @export(&mod.cpu.repack.q4_0.ggml_gemv_q4_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q4_0_4x4_q8_0_generic" });
    @export(&mod.cpu.repack.q4_0.ggml_gemv_q4_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q4_0_4x8_q8_0_generic" });
    @export(&mod.cpu.repack.q4_0.ggml_gemv_q4_0_8x8_q8_0_generic, .{ .name = "ggml_gemv_q4_0_8x8_q8_0_generic" });
    @export(&mod.cpu.repack.q4_0.ggml_gemm_q4_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q4_0_4x4_q8_0_generic" });
    @export(&mod.cpu.repack.q4_0.ggml_gemm_q4_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q4_0_4x8_q8_0_generic" });
    @export(&mod.cpu.repack.q4_0.ggml_gemm_q4_0_8x8_q8_0_generic, .{ .name = "ggml_gemm_q4_0_8x8_q8_0_generic" });
    // cpu/repack/q4_k.zig
    @export(&mod.cpu.repack.q4_k.ggml_gemv_q4_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q4_K_8x4_q8_K_generic" });
    @export(&mod.cpu.repack.q4_k.ggml_gemv_q4_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q4_K_8x8_q8_K_generic" });
    @export(&mod.cpu.repack.q4_k.ggml_gemm_q4_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q4_K_8x4_q8_K_generic" });
    @export(&mod.cpu.repack.q4_k.ggml_gemm_q4_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q4_K_8x8_q8_K_generic" });
    // cpu/repack/q5_k.zig
    @export(&mod.cpu.repack.q5_k.ggml_gemv_q5_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q5_K_8x4_q8_K_generic" });
    @export(&mod.cpu.repack.q5_k.ggml_gemv_q5_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q5_K_8x8_q8_K_generic" });
    @export(&mod.cpu.repack.q5_k.ggml_gemm_q5_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q5_K_8x4_q8_K_generic" });
    @export(&mod.cpu.repack.q5_k.ggml_gemm_q5_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q5_K_8x8_q8_K_generic" });
    // cpu/repack/q6_k.zig
    @export(&mod.cpu.repack.q6_k.ggml_gemv_q6_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q6_K_8x4_q8_K_generic" });
    @export(&mod.cpu.repack.q6_k.ggml_gemv_q6_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q6_K_8x8_q8_K_generic" });
    @export(&mod.cpu.repack.q6_k.ggml_gemm_q6_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q6_K_8x4_q8_K_generic" });
    @export(&mod.cpu.repack.q6_k.ggml_gemm_q6_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q6_K_8x8_q8_K_generic" });
    // cpu/repack/q8_0.zig
    @export(&mod.cpu.repack.q8_0.ggml_gemv_q8_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q8_0_4x4_q8_0_generic" });
    @export(&mod.cpu.repack.q8_0.ggml_gemv_q8_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q8_0_4x8_q8_0_generic" });
    @export(&mod.cpu.repack.q8_0.ggml_gemm_q8_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q8_0_4x4_q8_0_generic" });
    @export(&mod.cpu.repack.q8_0.ggml_gemm_q8_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q8_0_4x8_q8_0_generic" });
    // cpu/repack/quantize.zig
    @export(&mod.cpu.repack.quantize.ggml_quantize_mat_q8_0_4x4_generic, .{ .name = "ggml_quantize_mat_q8_0_4x4_generic" });
    @export(&mod.cpu.repack.quantize.ggml_quantize_mat_q8_0_4x8_generic, .{ .name = "ggml_quantize_mat_q8_0_4x8_generic" });
    // cpu/tensor.zig
    @export(&mod.cpu.tensor.ggml_new_i32, .{ .name = "ggml_new_i32" });
    @export(&mod.cpu.tensor.ggml_new_f32, .{ .name = "ggml_new_f32" });
    @export(&mod.cpu.tensor.ggml_set_i32, .{ .name = "ggml_set_i32" });
    @export(&mod.cpu.tensor.ggml_set_f32, .{ .name = "ggml_set_f32" });
    @export(&mod.cpu.tensor.ggml_get_i32_1d, .{ .name = "ggml_get_i32_1d" });
    @export(&mod.cpu.tensor.ggml_set_i32_1d, .{ .name = "ggml_set_i32_1d" });
    @export(&mod.cpu.tensor.ggml_get_f32_1d, .{ .name = "ggml_get_f32_1d" });
    @export(&mod.cpu.tensor.ggml_set_f32_1d, .{ .name = "ggml_set_f32_1d" });
    @export(&mod.cpu.tensor.ggml_get_i32_nd, .{ .name = "ggml_get_i32_nd" });
    @export(&mod.cpu.tensor.ggml_set_i32_nd, .{ .name = "ggml_set_i32_nd" });
    @export(&mod.cpu.tensor.ggml_get_f32_nd, .{ .name = "ggml_get_f32_nd" });
    @export(&mod.cpu.tensor.ggml_set_f32_nd, .{ .name = "ggml_set_f32_nd" });
    // cpu/threading.zig
    @export(&mod.cpu.threading.ggml_barrier, .{ .name = "ggml_barrier" });
    @export(&mod.cpu.threading.ggml_threadpool_chunk_set, .{ .name = "ggml_threadpool_chunk_set" });
    @export(&mod.cpu.threading.ggml_threadpool_chunk_add, .{ .name = "ggml_threadpool_chunk_add" });
    @export(&mod.cpu.threading.ggml_numa_init, .{ .name = "ggml_numa_init" });
    @export(&mod.cpu.threading.ggml_is_numa, .{ .name = "ggml_is_numa" });
    @export(&mod.cpu.threading.ggml_threadpool_new, .{ .name = "ggml_threadpool_new" });
    @export(&mod.cpu.threading.ggml_threadpool_free, .{ .name = "ggml_threadpool_free" });
    @export(&mod.cpu.threading.ggml_threadpool_pause, .{ .name = "ggml_threadpool_pause" });
    @export(&mod.cpu.threading.ggml_threadpool_resume, .{ .name = "ggml_threadpool_resume" });
    @export(&mod.cpu.threading.ggml_graph_compute, .{ .name = "ggml_graph_compute" });
    @export(&mod.cpu.threading.ggml_graph_compute_with_ctx, .{ .name = "ggml_graph_compute_with_ctx" });
    // cpu/tiled.zig
    @export(&mod.cpu.tiled.ggml_tiled_wdata_size, .{ .name = "ggml_tiled_wdata_size" });
    @export(&mod.cpu.tiled.ggml_compute_forward_mul_mat_tiled, .{ .name = "ggml_compute_forward_mul_mat_tiled" });
    @export(&mod.cpu.tiled.ggml_compute_forward_mul_mat_id_tiled, .{ .name = "ggml_compute_forward_mul_mat_id_tiled" });
    // cpu/traits.zig
    @export(&mod.cpu.traits.ggml_get_type_traits_cpu, .{ .name = "ggml_get_type_traits_cpu" });
    // cpu/unary_ops.zig
    @export(&mod.cpu.unary_ops.ggml_compute_forward_xielu, .{ .name = "ggml_compute_forward_xielu" });
    // cpu/vec.zig
    @export(&mod.cpu.vec.ggml_vec_dot_f32, .{ .name = "ggml_vec_dot_f32" });
    @export(&mod.cpu.vec.ggml_vec_dot_bf16, .{ .name = "ggml_vec_dot_bf16" });
    @export(&mod.cpu.vec.ggml_vec_dot_f16, .{ .name = "ggml_vec_dot_f16" });
    @export(&mod.cpu.vec.ggml_vec_dot_f16_f32, .{ .name = "ggml_vec_dot_f16_f32" });
    @export(&mod.cpu.vec.ggml_vec_silu_f32, .{ .name = "ggml_vec_silu_f32" });
    @export(&mod.cpu.vec.ggml_vec_swiglu_f32, .{ .name = "ggml_vec_swiglu_f32" });
    @export(&mod.cpu.vec.ggml_vec_gelu_quick_f32, .{ .name = "ggml_vec_gelu_quick_f32" });
    @export(&mod.cpu.vec.ggml_vec_geglu_quick_f32, .{ .name = "ggml_vec_geglu_quick_f32" });
    @export(&mod.cpu.vec.ggml_vec_cvar_f32, .{ .name = "ggml_vec_cvar_f32" });
    @export(&mod.cpu.vec.ggml_vec_soft_max_f32, .{ .name = "ggml_vec_soft_max_f32" });
    @export(&mod.cpu.vec.ggml_vec_log_soft_max_f32, .{ .name = "ggml_vec_log_soft_max_f32" });
    // gguf.zig
    @export(&mod.gguf.gguf_init_empty, .{ .name = "gguf_init_empty" });
    @export(&mod.gguf.gguf_init_from_callback, .{ .name = "gguf_init_from_callback" });
    @export(&mod.gguf.gguf_init_from_file_ptr, .{ .name = "gguf_init_from_file_ptr" });
    @export(&mod.gguf.gguf_init_from_buffer, .{ .name = "gguf_init_from_buffer" });
    @export(&mod.gguf.gguf_init_from_file, .{ .name = "gguf_init_from_file" });
    @export(&mod.gguf.gguf_free, .{ .name = "gguf_free" });
    @export(&mod.gguf.gguf_type_name, .{ .name = "gguf_type_name" });
    @export(&mod.gguf.gguf_get_version, .{ .name = "gguf_get_version" });
    @export(&mod.gguf.gguf_get_alignment, .{ .name = "gguf_get_alignment" });
    @export(&mod.gguf.gguf_get_data_offset, .{ .name = "gguf_get_data_offset" });
    @export(&mod.gguf.gguf_get_n_kv, .{ .name = "gguf_get_n_kv" });
    @export(&mod.gguf.gguf_find_key, .{ .name = "gguf_find_key" });
    @export(&mod.gguf.gguf_get_key, .{ .name = "gguf_get_key" });
    @export(&mod.gguf.gguf_get_kv_type, .{ .name = "gguf_get_kv_type" });
    @export(&mod.gguf.gguf_get_arr_type, .{ .name = "gguf_get_arr_type" });
    @export(&mod.gguf.gguf_get_arr_data, .{ .name = "gguf_get_arr_data" });
    @export(&mod.gguf.gguf_get_arr_str, .{ .name = "gguf_get_arr_str" });
    @export(&mod.gguf.gguf_get_arr_n, .{ .name = "gguf_get_arr_n" });
    @export(&mod.gguf.gguf_get_val_u8, .{ .name = "gguf_get_val_u8" });
    @export(&mod.gguf.gguf_get_val_i8, .{ .name = "gguf_get_val_i8" });
    @export(&mod.gguf.gguf_get_val_u16, .{ .name = "gguf_get_val_u16" });
    @export(&mod.gguf.gguf_get_val_i16, .{ .name = "gguf_get_val_i16" });
    @export(&mod.gguf.gguf_get_val_u32, .{ .name = "gguf_get_val_u32" });
    @export(&mod.gguf.gguf_get_val_i32, .{ .name = "gguf_get_val_i32" });
    @export(&mod.gguf.gguf_get_val_f32, .{ .name = "gguf_get_val_f32" });
    @export(&mod.gguf.gguf_get_val_u64, .{ .name = "gguf_get_val_u64" });
    @export(&mod.gguf.gguf_get_val_i64, .{ .name = "gguf_get_val_i64" });
    @export(&mod.gguf.gguf_get_val_f64, .{ .name = "gguf_get_val_f64" });
    @export(&mod.gguf.gguf_get_val_bool, .{ .name = "gguf_get_val_bool" });
    @export(&mod.gguf.gguf_get_val_str, .{ .name = "gguf_get_val_str" });
    @export(&mod.gguf.gguf_get_val_data, .{ .name = "gguf_get_val_data" });
    @export(&mod.gguf.gguf_get_n_tensors, .{ .name = "gguf_get_n_tensors" });
    @export(&mod.gguf.gguf_find_tensor, .{ .name = "gguf_find_tensor" });
    @export(&mod.gguf.gguf_get_tensor_offset, .{ .name = "gguf_get_tensor_offset" });
    @export(&mod.gguf.gguf_get_tensor_name, .{ .name = "gguf_get_tensor_name" });
    @export(&mod.gguf.gguf_get_tensor_ne, .{ .name = "gguf_get_tensor_ne" });
    @export(&mod.gguf.gguf_get_tensor_type, .{ .name = "gguf_get_tensor_type" });
    @export(&mod.gguf.gguf_get_tensor_size, .{ .name = "gguf_get_tensor_size" });
    @export(&mod.gguf.gguf_remove_key, .{ .name = "gguf_remove_key" });
    @export(&mod.gguf.gguf_set_val_u8, .{ .name = "gguf_set_val_u8" });
    @export(&mod.gguf.gguf_set_val_i8, .{ .name = "gguf_set_val_i8" });
    @export(&mod.gguf.gguf_set_val_u16, .{ .name = "gguf_set_val_u16" });
    @export(&mod.gguf.gguf_set_val_i16, .{ .name = "gguf_set_val_i16" });
    @export(&mod.gguf.gguf_set_val_u32, .{ .name = "gguf_set_val_u32" });
    @export(&mod.gguf.gguf_set_val_i32, .{ .name = "gguf_set_val_i32" });
    @export(&mod.gguf.gguf_set_val_f32, .{ .name = "gguf_set_val_f32" });
    @export(&mod.gguf.gguf_set_val_u64, .{ .name = "gguf_set_val_u64" });
    @export(&mod.gguf.gguf_set_val_i64, .{ .name = "gguf_set_val_i64" });
    @export(&mod.gguf.gguf_set_val_f64, .{ .name = "gguf_set_val_f64" });
    @export(&mod.gguf.gguf_set_val_bool, .{ .name = "gguf_set_val_bool" });
    @export(&mod.gguf.gguf_set_val_str, .{ .name = "gguf_set_val_str" });
    @export(&mod.gguf.gguf_set_arr_data, .{ .name = "gguf_set_arr_data" });
    @export(&mod.gguf.gguf_set_arr_str, .{ .name = "gguf_set_arr_str" });
    @export(&mod.gguf.gguf_set_kv, .{ .name = "gguf_set_kv" });
    @export(&mod.gguf.gguf_add_tensor, .{ .name = "gguf_add_tensor" });
    @export(&mod.gguf.gguf_set_tensor_type, .{ .name = "gguf_set_tensor_type" });
    @export(&mod.gguf.gguf_set_tensor_data, .{ .name = "gguf_set_tensor_data" });
    @export(&mod.gguf.gguf_write_to_file_ptr, .{ .name = "gguf_write_to_file_ptr" });
    @export(&mod.gguf.gguf_write_to_file, .{ .name = "gguf_write_to_file" });
    @export(&mod.gguf.gguf_get_meta_size, .{ .name = "gguf_get_meta_size" });
    @export(&mod.gguf.gguf_get_meta_data, .{ .name = "gguf_get_meta_data" });
    // graph.zig
    @export(&mod.graph.ggml_hash_size, .{ .name = "ggml_hash_size" });
    @export(&mod.graph.ggml_hash_set_new, .{ .name = "ggml_hash_set_new" });
    @export(&mod.graph.ggml_hash_set_reset, .{ .name = "ggml_hash_set_reset" });
    @export(&mod.graph.ggml_hash_set_free, .{ .name = "ggml_hash_set_free" });
    @export(&mod.graph.ggml_graph_overhead_custom, .{ .name = "ggml_graph_overhead_custom" });
    @export(&mod.graph.ggml_graph_overhead, .{ .name = "ggml_graph_overhead" });
    @export(&mod.graph.ggml_new_graph_custom, .{ .name = "ggml_new_graph_custom" });
    @export(&mod.graph.ggml_new_graph, .{ .name = "ggml_new_graph" });
    @export(&mod.graph.ggml_graph_view, .{ .name = "ggml_graph_view" });
    @export(&mod.graph.ggml_graph_cpy, .{ .name = "ggml_graph_cpy" });
    @export(&mod.graph.ggml_graph_dup, .{ .name = "ggml_graph_dup" });
    @export(&mod.graph.ggml_graph_reset, .{ .name = "ggml_graph_reset" });
    @export(&mod.graph.ggml_graph_clear, .{ .name = "ggml_graph_clear" });
    @export(&mod.graph.ggml_graph_size, .{ .name = "ggml_graph_size" });
    @export(&mod.graph.ggml_graph_node, .{ .name = "ggml_graph_node" });
    @export(&mod.graph.ggml_graph_nodes, .{ .name = "ggml_graph_nodes" });
    @export(&mod.graph.ggml_graph_n_nodes, .{ .name = "ggml_graph_n_nodes" });
    @export(&mod.graph.ggml_graph_add_node, .{ .name = "ggml_graph_add_node" });
    @export(&mod.graph.ggml_graph_get_tensor, .{ .name = "ggml_graph_get_tensor" });
    @export(&mod.graph.ggml_graph_get_grad, .{ .name = "ggml_graph_get_grad" });
    @export(&mod.graph.ggml_graph_get_grad_acc, .{ .name = "ggml_graph_get_grad_acc" });
    @export(&mod.graph.ggml_build_forward_select, .{ .name = "ggml_build_forward_select" });
    @export(&mod.graph.ggml_build_forward_expand, .{ .name = "ggml_build_forward_expand" });
    @export(&mod.graph.ggml_build_forward_order, .{ .name = "ggml_build_forward_order" });
    @export(&mod.graph.ggml_build_backward_expand, .{ .name = "ggml_build_backward_expand" });
    @export(&mod.graph.ggml_can_fuse_subgraph_ext, .{ .name = "ggml_can_fuse_subgraph_ext" });
    @export(&mod.graph.ggml_graph_print, .{ .name = "ggml_graph_print" });
    @export(&mod.graph.ggml_graph_dump_dot, .{ .name = "ggml_graph_dump_dot" });
    // ops.zig
    @export(&mod.ops.ggml_dup, .{ .name = "ggml_dup" });
    @export(&mod.ops.ggml_dup_inplace, .{ .name = "ggml_dup_inplace" });
    @export(&mod.ops.ggml_add, .{ .name = "ggml_add" });
    @export(&mod.ops.ggml_add_inplace, .{ .name = "ggml_add_inplace" });
    @export(&mod.ops.ggml_add_cast, .{ .name = "ggml_add_cast" });
    @export(&mod.ops.ggml_add_id, .{ .name = "ggml_add_id" });
    @export(&mod.ops.ggml_acc, .{ .name = "ggml_acc" });
    @export(&mod.ops.ggml_acc_inplace, .{ .name = "ggml_acc_inplace" });
    @export(&mod.ops.ggml_sub, .{ .name = "ggml_sub" });
    @export(&mod.ops.ggml_sub_inplace, .{ .name = "ggml_sub_inplace" });
    @export(&mod.ops.ggml_mul, .{ .name = "ggml_mul" });
    @export(&mod.ops.ggml_mul_inplace, .{ .name = "ggml_mul_inplace" });
    @export(&mod.ops.ggml_div, .{ .name = "ggml_div" });
    @export(&mod.ops.ggml_div_inplace, .{ .name = "ggml_div_inplace" });
    @export(&mod.ops.ggml_sqr, .{ .name = "ggml_sqr" });
    @export(&mod.ops.ggml_sqr_inplace, .{ .name = "ggml_sqr_inplace" });
    @export(&mod.ops.ggml_sqrt, .{ .name = "ggml_sqrt" });
    @export(&mod.ops.ggml_sqrt_inplace, .{ .name = "ggml_sqrt_inplace" });
    @export(&mod.ops.ggml_log, .{ .name = "ggml_log" });
    @export(&mod.ops.ggml_log_inplace, .{ .name = "ggml_log_inplace" });
    @export(&mod.ops.ggml_sin, .{ .name = "ggml_sin" });
    @export(&mod.ops.ggml_sin_inplace, .{ .name = "ggml_sin_inplace" });
    @export(&mod.ops.ggml_cos, .{ .name = "ggml_cos" });
    @export(&mod.ops.ggml_cos_inplace, .{ .name = "ggml_cos_inplace" });
    @export(&mod.ops.ggml_sum, .{ .name = "ggml_sum" });
    @export(&mod.ops.ggml_sum_rows, .{ .name = "ggml_sum_rows" });
    @export(&mod.ops.ggml_cumsum, .{ .name = "ggml_cumsum" });
    @export(&mod.ops.ggml_mean, .{ .name = "ggml_mean" });
    @export(&mod.ops.ggml_argmax, .{ .name = "ggml_argmax" });
    @export(&mod.ops.ggml_count_equal, .{ .name = "ggml_count_equal" });
    @export(&mod.ops.ggml_repeat, .{ .name = "ggml_repeat" });
    @export(&mod.ops.ggml_repeat_4d, .{ .name = "ggml_repeat_4d" });
    @export(&mod.ops.ggml_repeat_back, .{ .name = "ggml_repeat_back" });
    @export(&mod.ops.ggml_concat, .{ .name = "ggml_concat" });
    @export(&mod.ops.ggml_unary, .{ .name = "ggml_unary" });
    @export(&mod.ops.ggml_unary_inplace, .{ .name = "ggml_unary_inplace" });
    @export(&mod.ops.ggml_xielu, .{ .name = "ggml_xielu" });
    @export(&mod.ops.ggml_silu_back, .{ .name = "ggml_silu_back" });
    @export(&mod.ops.ggml_glu, .{ .name = "ggml_glu" });
    @export(&mod.ops.ggml_glu_split, .{ .name = "ggml_glu_split" });
    @export(&mod.ops.ggml_rms_norm_back, .{ .name = "ggml_rms_norm_back" });
    @export(&mod.ops.ggml_group_norm, .{ .name = "ggml_group_norm" });
    @export(&mod.ops.ggml_group_norm_inplace, .{ .name = "ggml_group_norm_inplace" });
    @export(&mod.ops.ggml_prec_set_acc, .{ .name = "ggml_prec_set_acc" });
    @export(&mod.ops.ggml_prec_set_src, .{ .name = "ggml_prec_set_src" });
    @export(&mod.ops.ggml_prec_get_acc, .{ .name = "ggml_prec_get_acc" });
    @export(&mod.ops.ggml_prec_get_src, .{ .name = "ggml_prec_get_src" });
    @export(&mod.ops.ggml_mul_mat, .{ .name = "ggml_mul_mat" });
    @export(&mod.ops.ggml_mul_mat_set_hint, .{ .name = "ggml_mul_mat_set_hint" });
    @export(&mod.ops.ggml_mul_mat_id, .{ .name = "ggml_mul_mat_id" });
    @export(&mod.ops.ggml_out_prod, .{ .name = "ggml_out_prod" });
    @export(&mod.ops.ggml_scale, .{ .name = "ggml_scale" });
    @export(&mod.ops.ggml_scale_inplace, .{ .name = "ggml_scale_inplace" });
    @export(&mod.ops.ggml_scale_bias, .{ .name = "ggml_scale_bias" });
    @export(&mod.ops.ggml_scale_bias_inplace, .{ .name = "ggml_scale_bias_inplace" });
    @export(&mod.ops.ggml_set, .{ .name = "ggml_set" });
    @export(&mod.ops.ggml_set_inplace, .{ .name = "ggml_set_inplace" });
    @export(&mod.ops.ggml_set_1d, .{ .name = "ggml_set_1d" });
    @export(&mod.ops.ggml_set_1d_inplace, .{ .name = "ggml_set_1d_inplace" });
    @export(&mod.ops.ggml_set_2d, .{ .name = "ggml_set_2d" });
    @export(&mod.ops.ggml_set_2d_inplace, .{ .name = "ggml_set_2d_inplace" });
    @export(&mod.ops.ggml_cpy, .{ .name = "ggml_cpy" });
    @export(&mod.ops.ggml_cast, .{ .name = "ggml_cast" });
    @export(&mod.ops.ggml_cont, .{ .name = "ggml_cont" });
    @export(&mod.ops.ggml_cont_4d, .{ .name = "ggml_cont_4d" });
    @export(&mod.ops.ggml_cont_1d, .{ .name = "ggml_cont_1d" });
    @export(&mod.ops.ggml_cont_2d, .{ .name = "ggml_cont_2d" });
    @export(&mod.ops.ggml_cont_3d, .{ .name = "ggml_cont_3d" });
    @export(&mod.ops.ggml_reshape, .{ .name = "ggml_reshape" });
    @export(&mod.ops.ggml_reshape_1d, .{ .name = "ggml_reshape_1d" });
    @export(&mod.ops.ggml_reshape_2d, .{ .name = "ggml_reshape_2d" });
    @export(&mod.ops.ggml_reshape_3d, .{ .name = "ggml_reshape_3d" });
    @export(&mod.ops.ggml_reshape_4d, .{ .name = "ggml_reshape_4d" });
    @export(&mod.ops.ggml_view_1d, .{ .name = "ggml_view_1d" });
    @export(&mod.ops.ggml_view_2d, .{ .name = "ggml_view_2d" });
    @export(&mod.ops.ggml_view_3d, .{ .name = "ggml_view_3d" });
    @export(&mod.ops.ggml_view_4d, .{ .name = "ggml_view_4d" });
    @export(&mod.ops.ggml_permute, .{ .name = "ggml_permute" });
    @export(&mod.ops.ggml_transpose, .{ .name = "ggml_transpose" });
    @export(&mod.ops.ggml_get_rows, .{ .name = "ggml_get_rows" });
    @export(&mod.ops.ggml_get_rows_back, .{ .name = "ggml_get_rows_back" });
    @export(&mod.ops.ggml_set_rows, .{ .name = "ggml_set_rows" });
    @export(&mod.ops.ggml_diag, .{ .name = "ggml_diag" });
    @export(&mod.ops.ggml_clamp, .{ .name = "ggml_clamp" });
    @export(&mod.ops.ggml_clamp_inplace, .{ .name = "ggml_clamp_inplace" });
    @export(&mod.ops.ggml_soft_max, .{ .name = "ggml_soft_max" });
    @export(&mod.ops.ggml_soft_max_inplace, .{ .name = "ggml_soft_max_inplace" });
    @export(&mod.ops.ggml_soft_max_ext, .{ .name = "ggml_soft_max_ext" });
    @export(&mod.ops.ggml_soft_max_ext_inplace, .{ .name = "ggml_soft_max_ext_inplace" });
    @export(&mod.ops.ggml_soft_max_add_sinks, .{ .name = "ggml_soft_max_add_sinks" });
    @export(&mod.ops.ggml_soft_max_ext_back, .{ .name = "ggml_soft_max_ext_back" });
    @export(&mod.ops.ggml_soft_max_ext_back_inplace, .{ .name = "ggml_soft_max_ext_back_inplace" });
    @export(&mod.ops.ggml_rope, .{ .name = "ggml_rope" });
    @export(&mod.ops.ggml_rope_inplace, .{ .name = "ggml_rope_inplace" });
    @export(&mod.ops.ggml_rope_ext, .{ .name = "ggml_rope_ext" });
    @export(&mod.ops.ggml_rope_ext_inplace, .{ .name = "ggml_rope_ext_inplace" });
    @export(&mod.ops.ggml_rope_multi, .{ .name = "ggml_rope_multi" });
    @export(&mod.ops.ggml_rope_multi_inplace, .{ .name = "ggml_rope_multi_inplace" });
    @export(&mod.ops.ggml_rope_ext_back, .{ .name = "ggml_rope_ext_back" });
    @export(&mod.ops.ggml_rope_multi_back, .{ .name = "ggml_rope_multi_back" });
    @export(&mod.ops.ggml_rope_set_offset, .{ .name = "ggml_rope_set_offset" });
    @export(&mod.ops.ggml_rope_yarn_corr_dims, .{ .name = "ggml_rope_yarn_corr_dims" });
    @export(&mod.ops.ggml_rope_set_truncate, .{ .name = "ggml_rope_set_truncate" });
    @export(&mod.ops.ggml_rope_get_truncate, .{ .name = "ggml_rope_get_truncate" });
    @export(&mod.ops.ggml_im2col, .{ .name = "ggml_im2col" });
    @export(&mod.ops.ggml_im2col_back, .{ .name = "ggml_im2col_back" });
    @export(&mod.ops.ggml_conv_1d, .{ .name = "ggml_conv_1d" });
    @export(&mod.ops.ggml_conv_1d_ph, .{ .name = "ggml_conv_1d_ph" });
    @export(&mod.ops.ggml_conv_1d_dw, .{ .name = "ggml_conv_1d_dw" });
    @export(&mod.ops.ggml_conv_1d_dw_ph, .{ .name = "ggml_conv_1d_dw_ph" });
    @export(&mod.ops.ggml_col2im_1d, .{ .name = "ggml_col2im_1d" });
    @export(&mod.ops.ggml_conv_transpose_1d, .{ .name = "ggml_conv_transpose_1d" });
    @export(&mod.ops.ggml_conv_2d, .{ .name = "ggml_conv_2d" });
    @export(&mod.ops.ggml_conv_2d_sk_p0, .{ .name = "ggml_conv_2d_sk_p0" });
    @export(&mod.ops.ggml_conv_2d_s1_ph, .{ .name = "ggml_conv_2d_s1_ph" });
    @export(&mod.ops.ggml_conv_2d_dw, .{ .name = "ggml_conv_2d_dw" });
    @export(&mod.ops.ggml_conv_2d_dw_direct, .{ .name = "ggml_conv_2d_dw_direct" });
    @export(&mod.ops.ggml_conv_2d_direct, .{ .name = "ggml_conv_2d_direct" });
    @export(&mod.ops.ggml_im2col_3d, .{ .name = "ggml_im2col_3d" });
    @export(&mod.ops.ggml_conv_3d, .{ .name = "ggml_conv_3d" });
    @export(&mod.ops.ggml_conv_3d_direct, .{ .name = "ggml_conv_3d_direct" });
    @export(&mod.ops.ggml_conv_transpose_2d_p0, .{ .name = "ggml_conv_transpose_2d_p0" });
    @export(&mod.ops.ggml_pool_1d, .{ .name = "ggml_pool_1d" });
    @export(&mod.ops.ggml_pool_2d, .{ .name = "ggml_pool_2d" });
    @export(&mod.ops.ggml_pool_2d_back, .{ .name = "ggml_pool_2d_back" });
    @export(&mod.ops.ggml_upscale, .{ .name = "ggml_upscale" });
    @export(&mod.ops.ggml_interpolate, .{ .name = "ggml_interpolate" });
    @export(&mod.ops.ggml_pad_ext, .{ .name = "ggml_pad_ext" });
    @export(&mod.ops.ggml_pad, .{ .name = "ggml_pad" });
    @export(&mod.ops.ggml_pad_ext_circular, .{ .name = "ggml_pad_ext_circular" });
    @export(&mod.ops.ggml_pad_circular, .{ .name = "ggml_pad_circular" });
    @export(&mod.ops.ggml_pad_reflect_1d, .{ .name = "ggml_pad_reflect_1d" });
    @export(&mod.ops.ggml_argsort, .{ .name = "ggml_argsort" });
    @export(&mod.ops.ggml_argsort_top_k, .{ .name = "ggml_argsort_top_k" });
    @export(&mod.ops.ggml_top_k, .{ .name = "ggml_top_k" });
    @export(&mod.ops.ggml_arange, .{ .name = "ggml_arange" });
    @export(&mod.ops.ggml_timestep_embedding, .{ .name = "ggml_timestep_embedding" });
    @export(&mod.ops.ggml_roll, .{ .name = "ggml_roll" });
    @export(&mod.ops.ggml_tri, .{ .name = "ggml_tri" });
    @export(&mod.ops.ggml_fill, .{ .name = "ggml_fill" });
    @export(&mod.ops.ggml_fill_inplace, .{ .name = "ggml_fill_inplace" });
    @export(&mod.ops.ggml_leaky_relu, .{ .name = "ggml_leaky_relu" });
    @export(&mod.ops.ggml_swiglu_oai, .{ .name = "ggml_swiglu_oai" });
    @export(&mod.ops.ggml_swiglu_clamp, .{ .name = "ggml_swiglu_clamp" });
    @export(&mod.ops.ggml_win_part, .{ .name = "ggml_win_part" });
    @export(&mod.ops.ggml_win_unpart, .{ .name = "ggml_win_unpart" });
    @export(&mod.ops.ggml_get_rel_pos, .{ .name = "ggml_get_rel_pos" });
    @export(&mod.ops.ggml_add_rel_pos, .{ .name = "ggml_add_rel_pos" });
    @export(&mod.ops.ggml_add_rel_pos_inplace, .{ .name = "ggml_add_rel_pos_inplace" });
    @export(&mod.ops.ggml_set_input, .{ .name = "ggml_set_input" });
    @export(&mod.ops.ggml_set_output, .{ .name = "ggml_set_output" });
    @export(&mod.ops.ggml_set_param, .{ .name = "ggml_set_param" });
    @export(&mod.ops.ggml_set_loss, .{ .name = "ggml_set_loss" });
    @export(&mod.ops.ggml_set_zero, .{ .name = "ggml_set_zero" });
    @export(&mod.ops.ggml_flash_attn_ext, .{ .name = "ggml_flash_attn_ext" });
    @export(&mod.ops.ggml_flash_attn_ext_rows, .{ .name = "ggml_flash_attn_ext_rows" });
    @export(&mod.ops.ggml_flash_attn_ext_get_prec, .{ .name = "ggml_flash_attn_ext_get_prec" });
    @export(&mod.ops.ggml_flash_attn_ext_set_n_kv_max, .{ .name = "ggml_flash_attn_ext_set_n_kv_max" });
    @export(&mod.ops.ggml_flash_attn_ext_add_sinks, .{ .name = "ggml_flash_attn_ext_add_sinks" });
    @export(&mod.ops.ggml_flash_attn_back, .{ .name = "ggml_flash_attn_back" });
    @export(&mod.ops.ggml_ssm_conv, .{ .name = "ggml_ssm_conv" });
    @export(&mod.ops.ggml_ssm_scan, .{ .name = "ggml_ssm_scan" });
    @export(&mod.ops.ggml_rwkv_wkv6, .{ .name = "ggml_rwkv_wkv6" });
    @export(&mod.ops.ggml_gated_linear_attn, .{ .name = "ggml_gated_linear_attn" });
    @export(&mod.ops.ggml_rwkv_wkv7, .{ .name = "ggml_rwkv_wkv7" });
    @export(&mod.ops.ggml_gated_delta_net, .{ .name = "ggml_gated_delta_net" });
    @export(&mod.ops.ggml_lightning_indexer, .{ .name = "ggml_lightning_indexer" });
    @export(&mod.ops.ggml_dsv4_hc_comb, .{ .name = "ggml_dsv4_hc_comb" });
    @export(&mod.ops.ggml_dsv4_hc_pre, .{ .name = "ggml_dsv4_hc_pre" });
    @export(&mod.ops.ggml_dsv4_hc_pre_gated, .{ .name = "ggml_dsv4_hc_pre_gated" });
    @export(&mod.ops.ggml_dsv4_hc_post, .{ .name = "ggml_dsv4_hc_post" });
    @export(&mod.ops.ggml_top_k_mask, .{ .name = "ggml_top_k_mask" });
    @export(&mod.ops.ggml_map_custom1, .{ .name = "ggml_map_custom1" });
    @export(&mod.ops.ggml_map_custom1_inplace, .{ .name = "ggml_map_custom1_inplace" });
    @export(&mod.ops.ggml_map_custom2, .{ .name = "ggml_map_custom2" });
    @export(&mod.ops.ggml_map_custom2_inplace, .{ .name = "ggml_map_custom2_inplace" });
    @export(&mod.ops.ggml_map_custom3, .{ .name = "ggml_map_custom3" });
    @export(&mod.ops.ggml_map_custom3_inplace, .{ .name = "ggml_map_custom3_inplace" });
    @export(&mod.ops.ggml_custom_4d, .{ .name = "ggml_custom_4d" });
    @export(&mod.ops.ggml_custom_inplace, .{ .name = "ggml_custom_inplace" });
    @export(&mod.ops.ggml_cross_entropy_loss, .{ .name = "ggml_cross_entropy_loss" });
    @export(&mod.ops.ggml_cross_entropy_loss_back, .{ .name = "ggml_cross_entropy_loss_back" });
    @export(&mod.ops.ggml_opt_step_adamw, .{ .name = "ggml_opt_step_adamw" });
    @export(&mod.ops.ggml_opt_step_sgd, .{ .name = "ggml_opt_step_sgd" });
    @export(&mod.ops.ggml_solve_tri, .{ .name = "ggml_solve_tri" });
    // quantize.zig
    @export(&mod.quantize.ggml_quantize_init, .{ .name = "ggml_quantize_init" });
    @export(&mod.quantize.ggml_quantize_free, .{ .name = "ggml_quantize_free" });
    @export(&mod.quantize.ggml_quantize_requires_imatrix, .{ .name = "ggml_quantize_requires_imatrix" });
    @export(&mod.quantize.ggml_quantize_chunk, .{ .name = "ggml_quantize_chunk" });
    // quants/chunks.zig
    @export(&mod.quants.chunks.quantize_q1_0, .{ .name = "quantize_q1_0" });
    @export(&mod.quants.chunks.quantize_q2_0, .{ .name = "quantize_q2_0" });
    // quants/codebook.zig
    @export(&mod.quants.codebook.iq2xs_init_impl, .{ .name = "iq2xs_init_impl" });
    @export(&mod.quants.codebook.iq2xs_free_impl, .{ .name = "iq2xs_free_impl" });
    @export(&mod.quants.codebook.iq3xs_init_impl, .{ .name = "iq3xs_init_impl" });
    @export(&mod.quants.codebook.iq3xs_free_impl, .{ .name = "iq3xs_free_impl" });
    // quants/iq1.zig
    @export(&mod.quants.iq1.quantize_iq1_s, .{ .name = "quantize_iq1_s" });
    @export(&mod.quants.iq1.quantize_iq1_m, .{ .name = "quantize_iq1_m" });
    // quants/iq2.zig
    @export(&mod.quants.iq2.quantize_iq2_xxs, .{ .name = "quantize_iq2_xxs" });
    @export(&mod.quants.iq2.quantize_iq2_xs, .{ .name = "quantize_iq2_xs" });
    @export(&mod.quants.iq2.quantize_iq2_s, .{ .name = "quantize_iq2_s" });
    @export(&mod.quants.iq2.quantize_row_iq2_s_ref, .{ .name = "quantize_row_iq2_s_ref" });
    // quants/iq3.zig
    @export(&mod.quants.iq3.quantize_iq3_xxs, .{ .name = "quantize_iq3_xxs" });
    @export(&mod.quants.iq3.quantize_row_iq3_xxs_ref, .{ .name = "quantize_row_iq3_xxs_ref" });
    @export(&mod.quants.iq3.quantize_iq3_s, .{ .name = "quantize_iq3_s" });
    @export(&mod.quants.iq3.quantize_row_iq3_s_ref, .{ .name = "quantize_row_iq3_s_ref" });
    // quants/iq4.zig
    @export(&mod.quants.iq4.quantize_row_iq4_nl_ref, .{ .name = "quantize_row_iq4_nl_ref" });
    @export(&mod.quants.iq4.quantize_iq4_nl, .{ .name = "quantize_iq4_nl" });
    @export(&mod.quants.iq4.quantize_iq4_xs, .{ .name = "quantize_iq4_xs" });
    @export(&mod.quants.iq4.quantize_row_iq4_xs_ref, .{ .name = "quantize_row_iq4_xs_ref" });
    // quants/iq_dequant.zig
    @export(&mod.quants.iq_dequant.dequantize_row_iq2_xxs, .{ .name = "dequantize_row_iq2_xxs" });
    @export(&mod.quants.iq_dequant.dequantize_row_iq2_xs, .{ .name = "dequantize_row_iq2_xs" });
    @export(&mod.quants.iq_dequant.dequantize_row_iq2_s, .{ .name = "dequantize_row_iq2_s" });
    @export(&mod.quants.iq_dequant.dequantize_row_iq3_xxs, .{ .name = "dequantize_row_iq3_xxs" });
    @export(&mod.quants.iq_dequant.dequantize_row_iq3_s, .{ .name = "dequantize_row_iq3_s" });
    @export(&mod.quants.iq_dequant.dequantize_row_iq1_s, .{ .name = "dequantize_row_iq1_s" });
    @export(&mod.quants.iq_dequant.dequantize_row_iq1_m, .{ .name = "dequantize_row_iq1_m" });
    @export(&mod.quants.iq_dequant.dequantize_row_iq4_nl, .{ .name = "dequantize_row_iq4_nl" });
    @export(&mod.quants.iq_dequant.dequantize_row_iq4_xs, .{ .name = "dequantize_row_iq4_xs" });
    // quants/k.zig
    @export(&mod.quants.k.quantize_row_q2_K_ref, .{ .name = "quantize_row_q2_K_ref" });
    @export(&mod.quants.k.dequantize_row_q2_K, .{ .name = "dequantize_row_q2_K" });
    @export(&mod.quants.k.quantize_row_q3_K_ref, .{ .name = "quantize_row_q3_K_ref" });
    @export(&mod.quants.k.dequantize_row_q3_K, .{ .name = "dequantize_row_q3_K" });
    @export(&mod.quants.k.quantize_row_q4_K_ref, .{ .name = "quantize_row_q4_K_ref" });
    @export(&mod.quants.k.dequantize_row_q4_K, .{ .name = "dequantize_row_q4_K" });
    @export(&mod.quants.k.quantize_row_q5_K_ref, .{ .name = "quantize_row_q5_K_ref" });
    @export(&mod.quants.k.dequantize_row_q5_K, .{ .name = "dequantize_row_q5_K" });
    @export(&mod.quants.k.quantize_row_q6_K_ref, .{ .name = "quantize_row_q6_K_ref" });
    @export(&mod.quants.k.dequantize_row_q6_K, .{ .name = "dequantize_row_q6_K" });
    @export(&mod.quants.k.quantize_row_q8_K_ref, .{ .name = "quantize_row_q8_K_ref" });
    @export(&mod.quants.k.dequantize_row_q8_K, .{ .name = "dequantize_row_q8_K" });
    // quants/legacy.zig
    @export(&mod.quants.legacy.quantize_row_q1_0_ref, .{ .name = "quantize_row_q1_0_ref" });
    @export(&mod.quants.legacy.dequantize_row_q1_0, .{ .name = "dequantize_row_q1_0" });
    @export(&mod.quants.legacy.quantize_row_q2_0_ref, .{ .name = "quantize_row_q2_0_ref" });
    @export(&mod.quants.legacy.dequantize_row_q2_0, .{ .name = "dequantize_row_q2_0" });
    @export(&mod.quants.legacy.quantize_row_q4_0_ref, .{ .name = "quantize_row_q4_0_ref" });
    @export(&mod.quants.legacy.dequantize_row_q4_0, .{ .name = "dequantize_row_q4_0" });
    @export(&mod.quants.legacy.quantize_row_q4_1_ref, .{ .name = "quantize_row_q4_1_ref" });
    @export(&mod.quants.legacy.dequantize_row_q4_1, .{ .name = "dequantize_row_q4_1" });
    @export(&mod.quants.legacy.quantize_row_q5_0_ref, .{ .name = "quantize_row_q5_0_ref" });
    @export(&mod.quants.legacy.dequantize_row_q5_0, .{ .name = "dequantize_row_q5_0" });
    @export(&mod.quants.legacy.quantize_row_q5_1_ref, .{ .name = "quantize_row_q5_1_ref" });
    @export(&mod.quants.legacy.dequantize_row_q5_1, .{ .name = "dequantize_row_q5_1" });
    @export(&mod.quants.legacy.quantize_row_q8_0_ref, .{ .name = "quantize_row_q8_0_ref" });
    @export(&mod.quants.legacy.dequantize_row_q8_0, .{ .name = "dequantize_row_q8_0" });
    @export(&mod.quants.legacy.quantize_row_q8_1_ref, .{ .name = "quantize_row_q8_1_ref" });
    @export(&mod.quants.legacy.quantize_row_mxfp4_ref, .{ .name = "quantize_row_mxfp4_ref" });
    @export(&mod.quants.legacy.dequantize_row_mxfp4, .{ .name = "dequantize_row_mxfp4" });
    @export(&mod.quants.legacy.quantize_row_nvfp4_ref, .{ .name = "quantize_row_nvfp4_ref" });
    @export(&mod.quants.legacy.dequantize_row_nvfp4, .{ .name = "dequantize_row_nvfp4" });
    @export(&mod.quants.legacy.quantize_row_f8_e4m3_ref, .{ .name = "quantize_row_f8_e4m3_ref" });
    @export(&mod.quants.legacy.dequantize_row_f8_e4m3, .{ .name = "dequantize_row_f8_e4m3" });
    // quants/ternary.zig
    @export(&mod.quants.ternary.quantize_row_tq1_0_ref, .{ .name = "quantize_row_tq1_0_ref" });
    @export(&mod.quants.ternary.dequantize_row_tq1_0, .{ .name = "dequantize_row_tq1_0" });
    @export(&mod.quants.ternary.quantize_row_tq2_0_ref, .{ .name = "quantize_row_tq2_0_ref" });
    @export(&mod.quants.ternary.dequantize_row_tq2_0, .{ .name = "dequantize_row_tq2_0" });
    // quants/validate.zig
    @export(&mod.quants.validate.ggml_validate_row_data, .{ .name = "ggml_validate_row_data" });
    // runtime.zig
    @export(&mod.runtime.ggml_graph_next_uid, .{ .name = "ggml_graph_next_uid" });
    @export(&mod.runtime.ggml_print_backtrace, .{ .name = "ggml_print_backtrace" });
    @export(&mod.runtime.ggml_set_abort_callback, .{ .name = "ggml_set_abort_callback" });
    @export(&mod.runtime.ggml_log_callback_default, .{ .name = "ggml_log_callback_default" });
    @export(&mod.runtime.ggml_log_get, .{ .name = "ggml_log_get" });
    @export(&mod.runtime.ggml_log_set, .{ .name = "ggml_log_set" });
    @export(&mod.runtime.ggml_aligned_malloc, .{ .name = "ggml_aligned_malloc" });
    @export(&mod.runtime.ggml_aligned_free, .{ .name = "ggml_aligned_free" });
    @export(&mod.runtime.ggml_status_to_string, .{ .name = "ggml_status_to_string" });
    @export(&mod.runtime.ggml_version, .{ .name = "ggml_version" });
    @export(&mod.runtime.ggml_commit, .{ .name = "ggml_commit" });
    @export(&mod.runtime.ggml_guid_matches, .{ .name = "ggml_guid_matches" });
    @export(&mod.runtime.ggml_fp16_to_fp32, .{ .name = "ggml_fp16_to_fp32" });
    @export(&mod.runtime.ggml_fp32_to_fp16, .{ .name = "ggml_fp32_to_fp16" });
    @export(&mod.runtime.ggml_bf16_to_fp32, .{ .name = "ggml_bf16_to_fp32" });
    @export(&mod.runtime.ggml_fp32_to_bf16, .{ .name = "ggml_fp32_to_bf16" });
    @export(&mod.runtime.ggml_fp16_to_fp32_row, .{ .name = "ggml_fp16_to_fp32_row" });
    @export(&mod.runtime.ggml_fp32_to_fp16_row, .{ .name = "ggml_fp32_to_fp16_row" });
    @export(&mod.runtime.ggml_bf16_to_fp32_row, .{ .name = "ggml_bf16_to_fp32_row" });
    @export(&mod.runtime.ggml_fp32_to_bf16_row_ref, .{ .name = "ggml_fp32_to_bf16_row_ref" });
    @export(&mod.runtime.ggml_fp32_to_bf16_row, .{ .name = "ggml_fp32_to_bf16_row" });
    @export(&mod.runtime.ggml_time_init, .{ .name = "ggml_time_init" });
    @export(&mod.runtime.ggml_time_ms, .{ .name = "ggml_time_ms" });
    @export(&mod.runtime.ggml_time_us, .{ .name = "ggml_time_us" });
    @export(&mod.runtime.ggml_cycles, .{ .name = "ggml_cycles" });
    @export(&mod.runtime.ggml_cycles_per_ms, .{ .name = "ggml_cycles_per_ms" });
    @export(&mod.runtime.ggml_threadpool_params_init, .{ .name = "ggml_threadpool_params_init" });
    @export(&mod.runtime.ggml_threadpool_params_default, .{ .name = "ggml_threadpool_params_default" });
    @export(&mod.runtime.ggml_threadpool_params_match, .{ .name = "ggml_threadpool_params_match" });
    @export(&mod.runtime.ggml_fopen, .{ .name = "ggml_fopen" });
    // threading.zig
    @export(&mod.threading.ggml_critical_section_start, .{ .name = "ggml_critical_section_start" });
    @export(&mod.threading.ggml_critical_section_end, .{ .name = "ggml_critical_section_end" });
    // types.zig
    @export(&mod.types.ggml_get_type_traits, .{ .name = "ggml_get_type_traits" });
    @export(&mod.types.ggml_blck_size, .{ .name = "ggml_blck_size" });
    @export(&mod.types.ggml_type_size, .{ .name = "ggml_type_size" });
    @export(&mod.types.ggml_row_size, .{ .name = "ggml_row_size" });
    @export(&mod.types.ggml_type_name, .{ .name = "ggml_type_name" });
    @export(&mod.types.ggml_is_quantized, .{ .name = "ggml_is_quantized" });
    @export(&mod.types.ggml_op_name, .{ .name = "ggml_op_name" });
    @export(&mod.types.ggml_op_symbol, .{ .name = "ggml_op_symbol" });
    @export(&mod.types.ggml_unary_op_name, .{ .name = "ggml_unary_op_name" });
    @export(&mod.types.ggml_glu_op_name, .{ .name = "ggml_glu_op_name" });
    @export(&mod.types.ggml_op_desc, .{ .name = "ggml_op_desc" });
    @export(&mod.types.ggml_get_unary_op, .{ .name = "ggml_get_unary_op" });
    @export(&mod.types.ggml_get_glu_op, .{ .name = "ggml_get_glu_op" });
    @export(&mod.types.ggml_ftype_to_ggml_type, .{ .name = "ggml_ftype_to_ggml_type" });
    @export(&mod.types.ggml_nelements, .{ .name = "ggml_nelements" });
    @export(&mod.types.ggml_nrows, .{ .name = "ggml_nrows" });
    @export(&mod.types.ggml_nbytes, .{ .name = "ggml_nbytes" });
    @export(&mod.types.ggml_nbytes_pad, .{ .name = "ggml_nbytes_pad" });
    @export(&mod.types.ggml_element_size, .{ .name = "ggml_element_size" });
    @export(&mod.types.ggml_tensor_overhead, .{ .name = "ggml_tensor_overhead" });
    @export(&mod.types.ggml_is_scalar, .{ .name = "ggml_is_scalar" });
    @export(&mod.types.ggml_is_vector, .{ .name = "ggml_is_vector" });
    @export(&mod.types.ggml_is_matrix, .{ .name = "ggml_is_matrix" });
    @export(&mod.types.ggml_is_3d, .{ .name = "ggml_is_3d" });
    @export(&mod.types.ggml_n_dims, .{ .name = "ggml_n_dims" });
    @export(&mod.types.ggml_is_empty, .{ .name = "ggml_is_empty" });
    @export(&mod.types.ggml_are_same_shape, .{ .name = "ggml_are_same_shape" });
    @export(&mod.types.ggml_are_same_stride, .{ .name = "ggml_are_same_stride" });
    @export(&mod.types.ggml_is_view, .{ .name = "ggml_is_view" });
    @export(&mod.types.ggml_can_repeat, .{ .name = "ggml_can_repeat" });
    @export(&mod.types.ggml_is_transposed, .{ .name = "ggml_is_transposed" });
    @export(&mod.types.ggml_is_contiguous, .{ .name = "ggml_is_contiguous" });
    @export(&mod.types.ggml_is_contiguous_0, .{ .name = "ggml_is_contiguous_0" });
    @export(&mod.types.ggml_is_contiguous_1, .{ .name = "ggml_is_contiguous_1" });
    @export(&mod.types.ggml_is_contiguous_2, .{ .name = "ggml_is_contiguous_2" });
    @export(&mod.types.ggml_is_contiguous_to_1, .{ .name = "ggml_is_contiguous_to_1" });
    @export(&mod.types.ggml_is_contiguous_to_2, .{ .name = "ggml_is_contiguous_to_2" });
    @export(&mod.types.ggml_is_contiguous_to_3, .{ .name = "ggml_is_contiguous_to_3" });
    @export(&mod.types.ggml_is_contiguously_allocated, .{ .name = "ggml_is_contiguously_allocated" });
    @export(&mod.types.ggml_is_permuted, .{ .name = "ggml_is_permuted" });
    @export(&mod.types.ggml_is_contiguous_channels, .{ .name = "ggml_is_contiguous_channels" });
    @export(&mod.types.ggml_is_contiguous_rows, .{ .name = "ggml_is_contiguous_rows" });
    if (config.blas_port) {
        // blas.zig
        @export(&mod.blas.ggml_backend_blas_init, .{ .name = "ggml_backend_blas_init" });
        @export(&mod.blas.ggml_backend_is_blas, .{ .name = "ggml_backend_is_blas" });
        @export(&mod.blas.ggml_backend_blas_set_n_threads, .{ .name = "ggml_backend_blas_set_n_threads" });
        @export(&mod.blas.ggml_backend_blas_reg, .{ .name = "ggml_backend_blas_reg" });
    }
    if (builtin.cpu.arch == .aarch64) {
        // cpu/quants/arm/iq.zig
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq2_xxs_q8_K, .{ .name = "ggml_vec_dot_iq2_xxs_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq2_xs_q8_K, .{ .name = "ggml_vec_dot_iq2_xs_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq2_s_q8_K, .{ .name = "ggml_vec_dot_iq2_s_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq3_xxs_q8_K, .{ .name = "ggml_vec_dot_iq3_xxs_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq3_s_q8_K, .{ .name = "ggml_vec_dot_iq3_s_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq1_s_q8_K, .{ .name = "ggml_vec_dot_iq1_s_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq1_m_q8_K, .{ .name = "ggml_vec_dot_iq1_m_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq4_xs_q8_K, .{ .name = "ggml_vec_dot_iq4_xs_q8_K" });
        // cpu/quants/arm/k.zig
        @export(&mod.cpu.quants.arch.k.ggml_vec_dot_q2_K_q8_K, .{ .name = "ggml_vec_dot_q2_K_q8_K" });
        @export(&mod.cpu.quants.arch.k.ggml_vec_dot_q3_K_q8_K, .{ .name = "ggml_vec_dot_q3_K_q8_K" });
        @export(&mod.cpu.quants.arch.k.ggml_vec_dot_q4_K_q8_K, .{ .name = "ggml_vec_dot_q4_K_q8_K" });
        @export(&mod.cpu.quants.arch.k.ggml_vec_dot_q5_K_q8_K, .{ .name = "ggml_vec_dot_q5_K_q8_K" });
        @export(&mod.cpu.quants.arch.k.ggml_vec_dot_q6_K_q8_K, .{ .name = "ggml_vec_dot_q6_K_q8_K" });
        // cpu/quants/arm/legacy.zig
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q1_0_q8_0, .{ .name = "ggml_vec_dot_q1_0_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q2_0_q8_0, .{ .name = "ggml_vec_dot_q2_0_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q4_0_q8_0, .{ .name = "ggml_vec_dot_q4_0_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q4_1_q8_1, .{ .name = "ggml_vec_dot_q4_1_q8_1" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_mxfp4_q8_0, .{ .name = "ggml_vec_dot_mxfp4_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_nvfp4_q8_0, .{ .name = "ggml_vec_dot_nvfp4_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q5_0_q8_0, .{ .name = "ggml_vec_dot_q5_0_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q5_1_q8_1, .{ .name = "ggml_vec_dot_q5_1_q8_1" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q8_0_q8_0, .{ .name = "ggml_vec_dot_q8_0_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_iq4_nl_q8_0, .{ .name = "ggml_vec_dot_iq4_nl_q8_0" });
        // cpu/quants/arm/rows.zig
        @export(&mod.cpu.quants.arch.rows.quantize_row_q8_0, .{ .name = "quantize_row_q8_0" });
        @export(&mod.cpu.quants.arch.rows.quantize_row_q8_1, .{ .name = "quantize_row_q8_1" });
        @export(&mod.cpu.quants.arch.rows.quantize_row_q8_K, .{ .name = "quantize_row_q8_K" });
        // cpu/quants/arm/ternary.zig
        @export(&mod.cpu.quants.arch.ternary.ggml_vec_dot_tq1_0_q8_K, .{ .name = "ggml_vec_dot_tq1_0_q8_K" });
        @export(&mod.cpu.quants.arch.ternary.ggml_vec_dot_tq2_0_q8_K, .{ .name = "ggml_vec_dot_tq2_0_q8_K" });
        // cpu/repack/arm/fallback.zig
        @export(&mod.cpu.repack.arch.fallback.ggml_gemv_q4_0_8x8_q8_0, .{ .name = "ggml_gemv_q4_0_8x8_q8_0" });
        @export(&mod.cpu.repack.arch.fallback.ggml_gemm_q4_0_4x8_q8_0, .{ .name = "ggml_gemm_q4_0_4x8_q8_0" });
        @export(&mod.cpu.repack.arch.fallback.ggml_gemm_q4_0_8x8_q8_0, .{ .name = "ggml_gemm_q4_0_8x8_q8_0" });
        @export(&mod.cpu.repack.arch.fallback.ggml_gemm_q4_K_8x8_q8_K, .{ .name = "ggml_gemm_q4_K_8x8_q8_K" });
        @export(&mod.cpu.repack.arch.fallback.ggml_gemm_q5_K_8x8_q8_K, .{ .name = "ggml_gemm_q5_K_8x8_q8_K" });
        @export(&mod.cpu.repack.arch.fallback.ggml_gemm_q6_K_8x8_q8_K, .{ .name = "ggml_gemm_q6_K_8x8_q8_K" });
        @export(&mod.cpu.repack.arch.fallback.ggml_gemm_q8_0_4x8_q8_0, .{ .name = "ggml_gemm_q8_0_4x8_q8_0" });
        // cpu/repack/arm/nibble4.zig
        @export(&mod.cpu.repack.arch.nibble4.ggml_gemv_iq4_nl_4x4_q8_0, .{ .name = "ggml_gemv_iq4_nl_4x4_q8_0" });
        @export(&mod.cpu.repack.arch.nibble4.ggml_gemv_mxfp4_4x4_q8_0, .{ .name = "ggml_gemv_mxfp4_4x4_q8_0" });
        @export(&mod.cpu.repack.arch.nibble4.ggml_gemm_iq4_nl_4x4_q8_0, .{ .name = "ggml_gemm_iq4_nl_4x4_q8_0" });
        @export(&mod.cpu.repack.arch.nibble4.ggml_gemm_mxfp4_4x4_q8_0, .{ .name = "ggml_gemm_mxfp4_4x4_q8_0" });
        // cpu/repack/arm/q1_0.zig
        @export(&mod.cpu.repack.arch.q1_0.ggml_gemv_q1_0_4x4_q8_0, .{ .name = "ggml_gemv_q1_0_4x4_q8_0" });
        @export(&mod.cpu.repack.arch.q1_0.ggml_gemv_q1_0_4x8_q8_0, .{ .name = "ggml_gemv_q1_0_4x8_q8_0" });
        @export(&mod.cpu.repack.arch.q1_0.ggml_gemm_q1_0_4x4_q8_0, .{ .name = "ggml_gemm_q1_0_4x4_q8_0" });
        @export(&mod.cpu.repack.arch.q1_0.ggml_gemm_q1_0_4x8_q8_0, .{ .name = "ggml_gemm_q1_0_4x8_q8_0" });
        // cpu/repack/arm/q4_0.zig
        @export(&mod.cpu.repack.arch.q4_0.ggml_gemv_q4_0_4x4_q8_0, .{ .name = "ggml_gemv_q4_0_4x4_q8_0" });
        @export(&mod.cpu.repack.arch.q4_0.ggml_gemv_q4_0_4x8_q8_0, .{ .name = "ggml_gemv_q4_0_4x8_q8_0" });
        // cpu/repack/arm/q4_0_gemm.zig
        @export(&mod.cpu.repack.arch.q4_0_gemm.ggml_gemm_q4_0_4x4_q8_0, .{ .name = "ggml_gemm_q4_0_4x4_q8_0" });
        // cpu/repack/arm/q4_k.zig
        @export(&mod.cpu.repack.arch.q4_k.ggml_gemv_q4_K_8x4_q8_K, .{ .name = "ggml_gemv_q4_K_8x4_q8_K" });
        @export(&mod.cpu.repack.arch.q4_k.ggml_gemv_q4_K_8x8_q8_K, .{ .name = "ggml_gemv_q4_K_8x8_q8_K" });
        @export(&mod.cpu.repack.arch.q4_k.ggml_gemm_q4_K_8x4_q8_K, .{ .name = "ggml_gemm_q4_K_8x4_q8_K" });
        // cpu/repack/arm/q5_k.zig
        @export(&mod.cpu.repack.arch.q5_k.ggml_gemv_q5_K_8x4_q8_K, .{ .name = "ggml_gemv_q5_K_8x4_q8_K" });
        @export(&mod.cpu.repack.arch.q5_k.ggml_gemv_q5_K_8x8_q8_K, .{ .name = "ggml_gemv_q5_K_8x8_q8_K" });
        @export(&mod.cpu.repack.arch.q5_k.ggml_gemm_q5_K_8x4_q8_K, .{ .name = "ggml_gemm_q5_K_8x4_q8_K" });
        // cpu/repack/arm/q6_k.zig
        @export(&mod.cpu.repack.arch.q6_k.ggml_gemv_q6_K_8x4_q8_K, .{ .name = "ggml_gemv_q6_K_8x4_q8_K" });
        @export(&mod.cpu.repack.arch.q6_k.ggml_gemv_q6_K_8x8_q8_K, .{ .name = "ggml_gemv_q6_K_8x8_q8_K" });
        @export(&mod.cpu.repack.arch.q6_k.ggml_gemm_q6_K_8x4_q8_K, .{ .name = "ggml_gemm_q6_K_8x4_q8_K" });
        // cpu/repack/arm/q8_0.zig
        @export(&mod.cpu.repack.arch.q8_0.ggml_gemv_q8_0_4x4_q8_0, .{ .name = "ggml_gemv_q8_0_4x4_q8_0" });
        @export(&mod.cpu.repack.arch.q8_0.ggml_gemv_q8_0_4x8_q8_0, .{ .name = "ggml_gemv_q8_0_4x8_q8_0" });
        @export(&mod.cpu.repack.arch.q8_0.ggml_gemm_q8_0_4x4_q8_0, .{ .name = "ggml_gemm_q8_0_4x4_q8_0" });
        // cpu/repack/arm/quantize.zig
        @export(&mod.cpu.repack.arch.quantize.ggml_quantize_mat_q8_0_4x4, .{ .name = "ggml_quantize_mat_q8_0_4x4" });
        @export(&mod.cpu.repack.arch.quantize.ggml_quantize_mat_q8_0_4x8, .{ .name = "ggml_quantize_mat_q8_0_4x8" });
    }
    if (builtin.cpu.arch == .x86_64) {
        // cpu/quants/x86/iq.zig
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq2_xxs_q8_K, .{ .name = "ggml_vec_dot_iq2_xxs_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq2_xs_q8_K, .{ .name = "ggml_vec_dot_iq2_xs_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq2_s_q8_K, .{ .name = "ggml_vec_dot_iq2_s_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq3_xxs_q8_K, .{ .name = "ggml_vec_dot_iq3_xxs_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq3_s_q8_K, .{ .name = "ggml_vec_dot_iq3_s_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq1_s_q8_K, .{ .name = "ggml_vec_dot_iq1_s_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq1_m_q8_K, .{ .name = "ggml_vec_dot_iq1_m_q8_K" });
        @export(&mod.cpu.quants.arch.iq.ggml_vec_dot_iq4_xs_q8_K, .{ .name = "ggml_vec_dot_iq4_xs_q8_K" });
        // cpu/quants/x86/k.zig
        @export(&mod.cpu.quants.arch.k.ggml_vec_dot_q2_K_q8_K, .{ .name = "ggml_vec_dot_q2_K_q8_K" });
        @export(&mod.cpu.quants.arch.k.ggml_vec_dot_q3_K_q8_K, .{ .name = "ggml_vec_dot_q3_K_q8_K" });
        @export(&mod.cpu.quants.arch.k.ggml_vec_dot_q4_K_q8_K, .{ .name = "ggml_vec_dot_q4_K_q8_K" });
        @export(&mod.cpu.quants.arch.k.ggml_vec_dot_q5_K_q8_K, .{ .name = "ggml_vec_dot_q5_K_q8_K" });
        @export(&mod.cpu.quants.arch.k.ggml_vec_dot_q6_K_q8_K, .{ .name = "ggml_vec_dot_q6_K_q8_K" });
        // cpu/quants/x86/legacy.zig
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q1_0_q8_0, .{ .name = "ggml_vec_dot_q1_0_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q2_0_q8_0, .{ .name = "ggml_vec_dot_q2_0_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q4_0_q8_0, .{ .name = "ggml_vec_dot_q4_0_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q4_1_q8_1, .{ .name = "ggml_vec_dot_q4_1_q8_1" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_mxfp4_q8_0, .{ .name = "ggml_vec_dot_mxfp4_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_nvfp4_q8_0, .{ .name = "ggml_vec_dot_nvfp4_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q5_0_q8_0, .{ .name = "ggml_vec_dot_q5_0_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q5_1_q8_1, .{ .name = "ggml_vec_dot_q5_1_q8_1" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_q8_0_q8_0, .{ .name = "ggml_vec_dot_q8_0_q8_0" });
        @export(&mod.cpu.quants.arch.legacy.ggml_vec_dot_iq4_nl_q8_0, .{ .name = "ggml_vec_dot_iq4_nl_q8_0" });
        // cpu/quants/x86/rows.zig
        @export(&mod.cpu.quants.arch.rows.quantize_row_q8_0, .{ .name = "quantize_row_q8_0" });
        @export(&mod.cpu.quants.arch.rows.quantize_row_q8_1, .{ .name = "quantize_row_q8_1" });
        @export(&mod.cpu.quants.arch.rows.quantize_row_q8_K, .{ .name = "quantize_row_q8_K" });
        // cpu/quants/x86/ternary.zig
        @export(&mod.cpu.quants.arch.ternary.ggml_vec_dot_tq1_0_q8_K, .{ .name = "ggml_vec_dot_tq1_0_q8_K" });
        @export(&mod.cpu.quants.arch.ternary.ggml_vec_dot_tq2_0_q8_K, .{ .name = "ggml_vec_dot_tq2_0_q8_K" });
        // cpu/repack/x86/q2_k.zig
        @export(&mod.cpu.repack.arch.q2_k.ggml_gemv_q2_K_8x8_q8_K, .{ .name = "ggml_gemv_q2_K_8x8_q8_K" });
        @export(&mod.cpu.repack.arch.q2_k.ggml_gemm_q2_K_8x8_q8_K, .{ .name = "ggml_gemm_q2_K_8x8_q8_K" });
        // cpu/repack/x86/q4_b32.zig
        @export(&mod.cpu.repack.arch.q4_b32.ggml_gemv_q4_0_8x8_q8_0, .{ .name = "ggml_gemv_q4_0_8x8_q8_0" });
        @export(&mod.cpu.repack.arch.q4_b32.ggml_gemv_iq4_nl_8x8_q8_0, .{ .name = "ggml_gemv_iq4_nl_8x8_q8_0" });
        @export(&mod.cpu.repack.arch.q4_b32.ggml_gemv_mxfp4_8x8_q8_0, .{ .name = "ggml_gemv_mxfp4_8x8_q8_0" });
        @export(&mod.cpu.repack.arch.q4_b32.ggml_gemm_q4_0_8x8_q8_0, .{ .name = "ggml_gemm_q4_0_8x8_q8_0" });
        @export(&mod.cpu.repack.arch.q4_b32.ggml_gemm_iq4_nl_8x8_q8_0, .{ .name = "ggml_gemm_iq4_nl_8x8_q8_0" });
        @export(&mod.cpu.repack.arch.q4_b32.ggml_gemm_mxfp4_8x8_q8_0, .{ .name = "ggml_gemm_mxfp4_8x8_q8_0" });
        // cpu/repack/x86/q4_k.zig
        @export(&mod.cpu.repack.arch.q4_k.ggml_gemv_q4_K_8x8_q8_K, .{ .name = "ggml_gemv_q4_K_8x8_q8_K" });
        @export(&mod.cpu.repack.arch.q4_k.ggml_gemm_q4_K_8x8_q8_K, .{ .name = "ggml_gemm_q4_K_8x8_q8_K" });
        // cpu/repack/x86/quantize.zig
        @export(&mod.cpu.repack.arch.quantize.ggml_quantize_mat_q8_0_4x8, .{ .name = "ggml_quantize_mat_q8_0_4x8" });
        @export(&mod.cpu.repack.arch.quantize.ggml_quantize_mat_q8_K_4x8, .{ .name = "ggml_quantize_mat_q8_K_4x8" });
    }
    if (config.use_metal) {
        // metal/backend.zig
        @export(&mod.metal.backend.ggml_backend_metal_init, .{ .name = "ggml_backend_metal_init" });
        @export(&mod.metal.backend.ggml_backend_is_metal, .{ .name = "ggml_backend_is_metal" });
        @export(&mod.metal.backend.ggml_backend_metal_set_abort_callback, .{ .name = "ggml_backend_metal_set_abort_callback" });
        @export(&mod.metal.backend.ggml_backend_metal_supports_family, .{ .name = "ggml_backend_metal_supports_family" });
        @export(&mod.metal.backend.ggml_backend_metal_capture_next_compute, .{ .name = "ggml_backend_metal_capture_next_compute" });
        @export(&mod.metal.backend.ggml_backend_metal_reg, .{ .name = "ggml_backend_metal_reg" });
        // metal/common.zig
        @export(&mod.metal.common.ggml_metal_op_mul_mat_use_fwht, .{ .name = "ggml_metal_op_mul_mat_use_fwht" });
        @export(&mod.metal.common.ggml_metal_op_mul_mat_use_mm, .{ .name = "ggml_metal_op_mul_mat_use_mm" });
        @export(&mod.metal.common.ggml_metal_op_mul_mat_id_use_mm, .{ .name = "ggml_metal_op_mul_mat_id_use_mm" });
        @export(&mod.metal.common.ggml_metal_mul_mv_mma_kind, .{ .name = "ggml_metal_mul_mv_mma_kind" });
        @export(&mod.metal.common.ggml_metal_mul_mv_mma_rt, .{ .name = "ggml_metal_mul_mv_mma_rt" });
        @export(&mod.metal.common.ggml_metal_mul_mv_mma_k_step, .{ .name = "ggml_metal_mul_mv_mma_k_step" });
        @export(&mod.metal.common.ggml_metal_op_mul_mat_use_mma, .{ .name = "ggml_metal_op_mul_mat_use_mma" });
        @export(&mod.metal.common.ggml_metal_op_mul_mat_may_use_mma, .{ .name = "ggml_metal_op_mul_mat_may_use_mma" });
        @export(&mod.metal.common.ggml_mem_ranges_init, .{ .name = "ggml_mem_ranges_init" });
        @export(&mod.metal.common.ggml_mem_ranges_free, .{ .name = "ggml_mem_ranges_free" });
        @export(&mod.metal.common.ggml_mem_ranges_reset, .{ .name = "ggml_mem_ranges_reset" });
        @export(&mod.metal.common.ggml_mem_ranges_add, .{ .name = "ggml_mem_ranges_add" });
        @export(&mod.metal.common.ggml_mem_ranges_check, .{ .name = "ggml_mem_ranges_check" });
        @export(&mod.metal.common.ggml_graph_optimize, .{ .name = "ggml_graph_optimize" });
        // metal/device_c.zig
        @export(&mod.metal.device_c.zz_props_sizeof, .{ .name = "zz_props_sizeof" });
        @export(&mod.metal.device_c.zz_pwp_sizeof, .{ .name = "zz_pwp_sizeof" });
        @export(&mod.metal.device_c.zz_pwp_nfields, .{ .name = "zz_pwp_nfields" });
        @export(&mod.metal.device_c.zz_pwp_offset, .{ .name = "zz_pwp_offset" });
        @export(&mod.metal.device_c.zz_pwp_field_size, .{ .name = "zz_pwp_field_size" });
        @export(&mod.metal.device_c.zz_pwp_roundtrip, .{ .name = "zz_pwp_roundtrip" });
        @export(&mod.metal.device_c.zz_props_alignof, .{ .name = "zz_props_alignof" });
        @export(&mod.metal.device_c.zz_props_nfields, .{ .name = "zz_props_nfields" });
        @export(&mod.metal.device_c.zz_props_offset, .{ .name = "zz_props_offset" });
        @export(&mod.metal.device_c.zz_props_field_size, .{ .name = "zz_props_field_size" });
        @export(&mod.metal.device_c.zz_bid_sizeof, .{ .name = "zz_bid_sizeof" });
        @export(&mod.metal.device_c.zz_bid_alignof, .{ .name = "zz_bid_alignof" });
        @export(&mod.metal.device_c.zz_bid_nfields, .{ .name = "zz_bid_nfields" });
        @export(&mod.metal.device_c.zz_bid_offset, .{ .name = "zz_bid_offset" });
        @export(&mod.metal.device_c.zz_bid_field_size, .{ .name = "zz_bid_field_size" });
        @export(&mod.metal.device_c.zz_bid_roundtrip, .{ .name = "zz_bid_roundtrip" });
        // metal/fusion.zig
        @export(&mod.metal.fusion.ggml_metal_fusion_get_id, .{ .name = "ggml_metal_fusion_get_id" });
        @export(&mod.metal.fusion.ggml_metal_fusion_add_alloc_deps, .{ .name = "ggml_metal_fusion_add_alloc_deps" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_init, .{ .name = "ggml_metal_fusion_info_init" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_free, .{ .name = "ggml_metal_fusion_info_free" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_enabled, .{ .name = "ggml_metal_fusion_info_enabled" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_stats, .{ .name = "ggml_metal_fusion_info_stats" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_debug, .{ .name = "ggml_metal_fusion_info_debug" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_n_fusions, .{ .name = "ggml_metal_fusion_info_n_fusions" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_label, .{ .name = "ggml_metal_fusion_info_label" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_count, .{ .name = "ggml_metal_fusion_info_count" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_count_fusion, .{ .name = "ggml_metal_fusion_info_count_fusion" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_set_enabled, .{ .name = "ggml_metal_fusion_info_set_enabled" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_labels_init, .{ .name = "ggml_metal_fusion_info_labels_init" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_stats_init, .{ .name = "ggml_metal_fusion_info_stats_init" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_stats_reset, .{ .name = "ggml_metal_fusion_info_stats_reset" });
        @export(&mod.metal.fusion.ggml_metal_fusion_info_stats_get, .{ .name = "ggml_metal_fusion_info_stats_get" });
        @export(&mod.metal.fusion.ggml_metal_fusion_next, .{ .name = "ggml_metal_fusion_next" });
        @export(&mod.metal.fusion.ggml_metal_fusion_max, .{ .name = "ggml_metal_fusion_max" });
        // metal/impl_c.zig
        @export(&mod.metal.impl_c.zz_fc_value, .{ .name = "zz_fc_value" });
        @export(&mod.metal.impl_c.zz_fc_name, .{ .name = "zz_fc_name" });
        @export(&mod.metal.impl_c.zz_fc_count, .{ .name = "zz_fc_count" });
        // metal/kargs.zig
        @export(&mod.metal.kargs.zz_kargs_count, .{ .name = "zz_kargs_count" });
        @export(&mod.metal.kargs.zz_kargs_name, .{ .name = "zz_kargs_name" });
        @export(&mod.metal.kargs.zz_kargs_sizeof, .{ .name = "zz_kargs_sizeof" });
        @export(&mod.metal.kargs.zz_kargs_alignof, .{ .name = "zz_kargs_alignof" });
        @export(&mod.metal.kargs.zz_kargs_nfields, .{ .name = "zz_kargs_nfields" });
        @export(&mod.metal.kargs.zz_kargs_field_offset, .{ .name = "zz_kargs_field_offset" });
        @export(&mod.metal.kargs.zz_kargs_field_size, .{ .name = "zz_kargs_field_size" });
        @export(&mod.metal.kargs.zz_kargs_field_name, .{ .name = "zz_kargs_field_name" });
        // metal/library.zig
        @export(&mod.metal.library.ggml_metal_device_get, .{ .name = "ggml_metal_device_get" });
        @export(&mod.metal.library.ggml_metal_pipelines_init, .{ .name = "ggml_metal_pipelines_init" });
        @export(&mod.metal.library.ggml_metal_pipelines_free, .{ .name = "ggml_metal_pipelines_free" });
        @export(&mod.metal.library.ggml_metal_pipelines_add, .{ .name = "ggml_metal_pipelines_add" });
        @export(&mod.metal.library.ggml_metal_pipelines_get, .{ .name = "ggml_metal_pipelines_get" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_base, .{ .name = "ggml_metal_library_get_pipeline_base" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_cpy, .{ .name = "ggml_metal_library_get_pipeline_cpy" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_pool_1d, .{ .name = "ggml_metal_library_get_pipeline_pool_1d" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_pool_2d, .{ .name = "ggml_metal_library_get_pipeline_pool_2d" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_get_rows, .{ .name = "ggml_metal_library_get_pipeline_get_rows" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_set_rows, .{ .name = "ggml_metal_library_get_pipeline_set_rows" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_diag, .{ .name = "ggml_metal_library_get_pipeline_diag" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_repeat, .{ .name = "ggml_metal_library_get_pipeline_repeat" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_concat, .{ .name = "ggml_metal_library_get_pipeline_concat" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_unary, .{ .name = "ggml_metal_library_get_pipeline_unary" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_glu, .{ .name = "ggml_metal_library_get_pipeline_glu" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_sum, .{ .name = "ggml_metal_library_get_pipeline_sum" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_sum_rows, .{ .name = "ggml_metal_library_get_pipeline_sum_rows" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_cumsum_blk, .{ .name = "ggml_metal_library_get_pipeline_cumsum_blk" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_cumsum_add, .{ .name = "ggml_metal_library_get_pipeline_cumsum_add" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_tri, .{ .name = "ggml_metal_library_get_pipeline_tri" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_soft_max, .{ .name = "ggml_metal_library_get_pipeline_soft_max" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_lightning_indexer, .{ .name = "ggml_metal_library_get_pipeline_lightning_indexer" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_top_k_mask, .{ .name = "ggml_metal_library_get_pipeline_top_k_mask" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_dsv4_hc, .{ .name = "ggml_metal_library_get_pipeline_dsv4_hc" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_ssm_conv, .{ .name = "ggml_metal_library_get_pipeline_ssm_conv" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_ssm_conv_batched, .{ .name = "ggml_metal_library_get_pipeline_ssm_conv_batched" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_ssm_scan, .{ .name = "ggml_metal_library_get_pipeline_ssm_scan" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_ssm_scan_ssd_mma, .{ .name = "ggml_metal_library_get_pipeline_ssm_scan_ssd_mma" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_rwkv, .{ .name = "ggml_metal_library_get_pipeline_rwkv" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_gated_delta_net, .{ .name = "ggml_metal_library_get_pipeline_gated_delta_net" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_solve_tri, .{ .name = "ggml_metal_library_get_pipeline_solve_tri" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_mul_mv_ext, .{ .name = "ggml_metal_library_get_pipeline_mul_mv_ext" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_mul_mv_mma_auto, .{ .name = "ggml_metal_library_get_pipeline_mul_mv_mma_auto" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_mul_mv_mma, .{ .name = "ggml_metal_library_get_pipeline_mul_mv_mma" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_mul_mm, .{ .name = "ggml_metal_library_get_pipeline_mul_mm" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_mul_mv, .{ .name = "ggml_metal_library_get_pipeline_mul_mv" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_mul_mm_id_map0, .{ .name = "ggml_metal_library_get_pipeline_mul_mm_id_map0" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_mul_mm_id_zero, .{ .name = "ggml_metal_library_get_pipeline_mul_mm_id_zero" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_mul_mm_id, .{ .name = "ggml_metal_library_get_pipeline_mul_mm_id" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_mul_mv_id, .{ .name = "ggml_metal_library_get_pipeline_mul_mv_id" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_argmax, .{ .name = "ggml_metal_library_get_pipeline_argmax" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_argmax_part, .{ .name = "ggml_metal_library_get_pipeline_argmax_part" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_argmax_final, .{ .name = "ggml_metal_library_get_pipeline_argmax_final" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_argsort, .{ .name = "ggml_metal_library_get_pipeline_argsort" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_argsort_merge, .{ .name = "ggml_metal_library_get_pipeline_argsort_merge" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_fwht, .{ .name = "ggml_metal_library_get_pipeline_fwht" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_top_k, .{ .name = "ggml_metal_library_get_pipeline_top_k" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_top_k_radix, .{ .name = "ggml_metal_library_get_pipeline_top_k_radix" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_top_k_merge, .{ .name = "ggml_metal_library_get_pipeline_top_k_merge" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_topk_moe, .{ .name = "ggml_metal_library_get_pipeline_topk_moe" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_moe_reduce, .{ .name = "ggml_metal_library_get_pipeline_moe_reduce" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_pad, .{ .name = "ggml_metal_library_get_pipeline_flash_attn_ext_pad" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16, .{ .name = "ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_blk, .{ .name = "ggml_metal_library_get_pipeline_flash_attn_ext_blk" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_tensor, .{ .name = "ggml_metal_library_get_pipeline_flash_attn_ext_tensor" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext, .{ .name = "ggml_metal_library_get_pipeline_flash_attn_ext" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_rows, .{ .name = "ggml_metal_library_get_pipeline_flash_attn_ext_rows" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx, .{ .name = "ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_vec, .{ .name = "ggml_metal_library_get_pipeline_flash_attn_ext_vec" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce, .{ .name = "ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_bin, .{ .name = "ggml_metal_library_get_pipeline_bin" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_bin_one, .{ .name = "ggml_metal_library_get_pipeline_bin_one" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_l2_norm, .{ .name = "ggml_metal_library_get_pipeline_l2_norm" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_group_norm, .{ .name = "ggml_metal_library_get_pipeline_group_norm" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_norm, .{ .name = "ggml_metal_library_get_pipeline_norm" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_norm_scale, .{ .name = "ggml_metal_library_get_pipeline_norm_scale" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_rope, .{ .name = "ggml_metal_library_get_pipeline_rope" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_im2col, .{ .name = "ggml_metal_library_get_pipeline_im2col" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_conv_transpose_1d, .{ .name = "ggml_metal_library_get_pipeline_conv_transpose_1d" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_col2im_1d, .{ .name = "ggml_metal_library_get_pipeline_col2im_1d" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_snake, .{ .name = "ggml_metal_library_get_pipeline_snake" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_conv_transpose_2d, .{ .name = "ggml_metal_library_get_pipeline_conv_transpose_2d" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_conv_2d, .{ .name = "ggml_metal_library_get_pipeline_conv_2d" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_conv_2d_dw, .{ .name = "ggml_metal_library_get_pipeline_conv_2d_dw" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_conv_3d, .{ .name = "ggml_metal_library_get_pipeline_conv_3d" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_upscale, .{ .name = "ggml_metal_library_get_pipeline_upscale" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_roll, .{ .name = "ggml_metal_library_get_pipeline_roll" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_pad, .{ .name = "ggml_metal_library_get_pipeline_pad" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_pad_reflect_1d, .{ .name = "ggml_metal_library_get_pipeline_pad_reflect_1d" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_arange, .{ .name = "ggml_metal_library_get_pipeline_arange" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_timestep_embedding, .{ .name = "ggml_metal_library_get_pipeline_timestep_embedding" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_opt_step_adamw, .{ .name = "ggml_metal_library_get_pipeline_opt_step_adamw" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_opt_step_sgd, .{ .name = "ggml_metal_library_get_pipeline_opt_step_sgd" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_silu_back, .{ .name = "ggml_metal_library_get_pipeline_silu_back" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_memset, .{ .name = "ggml_metal_library_get_pipeline_memset" });
        @export(&mod.metal.library.ggml_metal_library_get_pipeline_count_equal, .{ .name = "ggml_metal_library_get_pipeline_count_equal" });
        // metal/ops.zig
        @export(&mod.metal.ops.ggml_metal_op_init, .{ .name = "ggml_metal_op_init" });
        @export(&mod.metal.ops.ggml_metal_op_free, .{ .name = "ggml_metal_op_free" });
        @export(&mod.metal.ops.ggml_metal_op_n_nodes, .{ .name = "ggml_metal_op_n_nodes" });
        @export(&mod.metal.ops.ggml_metal_op_encode, .{ .name = "ggml_metal_op_encode" });
        @export(&mod.metal.ops.ggml_metal_op_concat, .{ .name = "ggml_metal_op_concat" });
        @export(&mod.metal.ops.ggml_metal_op_bin, .{ .name = "ggml_metal_op_bin" });
        @export(&mod.metal.ops.ggml_metal_op_add_id, .{ .name = "ggml_metal_op_add_id" });
        @export(&mod.metal.ops.ggml_metal_op_repeat, .{ .name = "ggml_metal_op_repeat" });
        @export(&mod.metal.ops.ggml_metal_op_acc, .{ .name = "ggml_metal_op_acc" });
        @export(&mod.metal.ops.ggml_metal_op_unary, .{ .name = "ggml_metal_op_unary" });
        @export(&mod.metal.ops.ggml_metal_op_silu_back, .{ .name = "ggml_metal_op_silu_back" });
        @export(&mod.metal.ops.ggml_metal_op_glu, .{ .name = "ggml_metal_op_glu" });
        @export(&mod.metal.ops.ggml_metal_op_sum, .{ .name = "ggml_metal_op_sum" });
        @export(&mod.metal.ops.ggml_metal_op_sum_rows, .{ .name = "ggml_metal_op_sum_rows" });
        @export(&mod.metal.ops.ggml_metal_op_cumsum, .{ .name = "ggml_metal_op_cumsum" });
        @export(&mod.metal.ops.ggml_metal_op_lightning_indexer, .{ .name = "ggml_metal_op_lightning_indexer" });
        @export(&mod.metal.ops.ggml_metal_op_top_k_mask, .{ .name = "ggml_metal_op_top_k_mask" });
        @export(&mod.metal.ops.ggml_metal_op_dsv4_hc, .{ .name = "ggml_metal_op_dsv4_hc" });
        @export(&mod.metal.ops.ggml_metal_op_soft_max, .{ .name = "ggml_metal_op_soft_max" });
        @export(&mod.metal.ops.ggml_metal_op_ssm_conv, .{ .name = "ggml_metal_op_ssm_conv" });
        @export(&mod.metal.ops.ggml_metal_op_ssm_scan, .{ .name = "ggml_metal_op_ssm_scan" });
        @export(&mod.metal.ops.ggml_metal_op_rwkv, .{ .name = "ggml_metal_op_rwkv" });
        @export(&mod.metal.ops.ggml_metal_op_gated_delta_net, .{ .name = "ggml_metal_op_gated_delta_net" });
        @export(&mod.metal.ops.ggml_metal_op_solve_tri, .{ .name = "ggml_metal_op_solve_tri" });
        @export(&mod.metal.ops.ggml_metal_op_mul_mat, .{ .name = "ggml_metal_op_mul_mat" });
        @export(&mod.metal.ops.ggml_metal_op_mul_mat_id, .{ .name = "ggml_metal_op_mul_mat_id" });
        @export(&mod.metal.ops.ggml_metal_op_get_rows, .{ .name = "ggml_metal_op_get_rows" });
        @export(&mod.metal.ops.ggml_metal_op_set_rows, .{ .name = "ggml_metal_op_set_rows" });
        @export(&mod.metal.ops.ggml_metal_op_diag, .{ .name = "ggml_metal_op_diag" });
        @export(&mod.metal.ops.ggml_metal_op_l2_norm, .{ .name = "ggml_metal_op_l2_norm" });
        @export(&mod.metal.ops.ggml_metal_op_group_norm, .{ .name = "ggml_metal_op_group_norm" });
        @export(&mod.metal.ops.ggml_metal_op_norm, .{ .name = "ggml_metal_op_norm" });
        @export(&mod.metal.ops.ggml_metal_op_rope, .{ .name = "ggml_metal_op_rope" });
        @export(&mod.metal.ops.ggml_metal_op_im2col, .{ .name = "ggml_metal_op_im2col" });
        @export(&mod.metal.ops.ggml_metal_op_conv_2d, .{ .name = "ggml_metal_op_conv_2d" });
        @export(&mod.metal.ops.ggml_metal_op_conv_2d_dw, .{ .name = "ggml_metal_op_conv_2d_dw" });
        @export(&mod.metal.ops.ggml_metal_op_conv_transpose_1d, .{ .name = "ggml_metal_op_conv_transpose_1d" });
        @export(&mod.metal.ops.ggml_metal_op_conv_transpose_2d, .{ .name = "ggml_metal_op_conv_transpose_2d" });
        @export(&mod.metal.ops.ggml_metal_op_col2im_1d, .{ .name = "ggml_metal_op_col2im_1d" });
        @export(&mod.metal.ops.ggml_metal_op_conv_3d, .{ .name = "ggml_metal_op_conv_3d" });
        @export(&mod.metal.ops.ggml_metal_op_upscale, .{ .name = "ggml_metal_op_upscale" });
        @export(&mod.metal.ops.ggml_metal_op_pad, .{ .name = "ggml_metal_op_pad" });
        @export(&mod.metal.ops.ggml_metal_op_pad_reflect_1d, .{ .name = "ggml_metal_op_pad_reflect_1d" });
        @export(&mod.metal.ops.ggml_metal_op_roll, .{ .name = "ggml_metal_op_roll" });
        @export(&mod.metal.ops.ggml_metal_op_arange, .{ .name = "ggml_metal_op_arange" });
        @export(&mod.metal.ops.ggml_metal_op_timestep_embedding, .{ .name = "ggml_metal_op_timestep_embedding" });
        @export(&mod.metal.ops.ggml_metal_op_argsort, .{ .name = "ggml_metal_op_argsort" });
        @export(&mod.metal.ops.ggml_metal_op_topk_moe, .{ .name = "ggml_metal_op_topk_moe" });
        @export(&mod.metal.ops.ggml_metal_op_moe_reduce, .{ .name = "ggml_metal_op_moe_reduce" });
        @export(&mod.metal.ops.ggml_metal_op_top_k, .{ .name = "ggml_metal_op_top_k" });
        @export(&mod.metal.ops.ggml_metal_op_tri, .{ .name = "ggml_metal_op_tri" });
        @export(&mod.metal.ops.ggml_metal_op_flash_attn_ext, .{ .name = "ggml_metal_op_flash_attn_ext" });
        @export(&mod.metal.ops.ggml_metal_op_set, .{ .name = "ggml_metal_op_set" });
        @export(&mod.metal.ops.ggml_metal_op_cpy, .{ .name = "ggml_metal_op_cpy" });
        @export(&mod.metal.ops.ggml_metal_op_pool_1d, .{ .name = "ggml_metal_op_pool_1d" });
        @export(&mod.metal.ops.ggml_metal_op_pool_2d, .{ .name = "ggml_metal_op_pool_2d" });
        @export(&mod.metal.ops.ggml_metal_op_argmax_extra, .{ .name = "ggml_metal_op_argmax_extra" });
        @export(&mod.metal.ops.ggml_metal_op_argmax, .{ .name = "ggml_metal_op_argmax" });
        @export(&mod.metal.ops.ggml_metal_op_opt_step_adamw, .{ .name = "ggml_metal_op_opt_step_adamw" });
        @export(&mod.metal.ops.ggml_metal_op_opt_step_sgd, .{ .name = "ggml_metal_op_opt_step_sgd" });
        @export(&mod.metal.ops.ggml_metal_op_count_equal, .{ .name = "ggml_metal_op_count_equal" });
        @export(&mod.metal.ops.ggml_metal_op_fwht, .{ .name = "ggml_metal_op_fwht" });
        @export(&mod.metal.ops.ggml_metal_op_snake_fused, .{ .name = "ggml_metal_op_snake_fused" });
        @export(&mod.metal.ops.ggml_metal_op_mul_mat_id_extra_tpe, .{ .name = "ggml_metal_op_mul_mat_id_extra_tpe" });
        @export(&mod.metal.ops.ggml_metal_op_mul_mat_id_extra_ids, .{ .name = "ggml_metal_op_mul_mat_id_extra_ids" });
        @export(&mod.metal.ops.ggml_metal_op_flash_attn_ext_use_vec, .{ .name = "ggml_metal_op_flash_attn_ext_use_vec" });
        @export(&mod.metal.ops.ggml_metal_op_flash_attn_ext_extra_pad, .{ .name = "ggml_metal_op_flash_attn_ext_extra_pad" });
        @export(&mod.metal.ops.ggml_metal_op_flash_attn_ext_extra_blk, .{ .name = "ggml_metal_op_flash_attn_ext_extra_blk" });
        @export(&mod.metal.ops.ggml_metal_op_flash_attn_ext_extra_tmp, .{ .name = "ggml_metal_op_flash_attn_ext_extra_tmp" });
        @export(&mod.metal.ops.ggml_metal_op_flash_attn_ext_extra_kv_f16, .{ .name = "ggml_metal_op_flash_attn_ext_extra_kv_f16" });
        @export(&mod.metal.ops.ggml_metal_op_flash_attn_ext_extra_idx, .{ .name = "ggml_metal_op_flash_attn_ext_extra_idx" });
        @export(&mod.metal.ops.ggml_metal_op_flash_attn_ext_extra_rows, .{ .name = "ggml_metal_op_flash_attn_ext_extra_rows" });
    }
    // context.zig
    if (mod.impl.zig_varargs) @export(&mod.context.formatName, .{ .name = "ggml_format_name" });
    // cpu/repack/arm/fallback.zig
    if (builtin.cpu.arch == .aarch64) {
        @export(&mod.cpu.repack.quantize.ggml_quantize_mat_q8_K_4x4_generic, .{ .name = "ggml_quantize_mat_q8_K_4x4" });
        @export(&mod.cpu.repack.quantize.ggml_quantize_mat_q8_K_4x8_generic, .{ .name = "ggml_quantize_mat_q8_K_4x8" });
        @export(&mod.cpu.repack.nibble4.ggml_gemv_iq4_nl_8x8_q8_0_generic, .{ .name = "ggml_gemv_iq4_nl_8x8_q8_0" });
        @export(&mod.cpu.repack.nibble4.ggml_gemv_mxfp4_8x8_q8_0_generic, .{ .name = "ggml_gemv_mxfp4_8x8_q8_0" });
        @export(&mod.cpu.repack.q2_k.ggml_gemv_q2_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q2_K_8x8_q8_K" });
        @export(&mod.cpu.repack.nibble4.ggml_gemm_iq4_nl_8x8_q8_0_generic, .{ .name = "ggml_gemm_iq4_nl_8x8_q8_0" });
        @export(&mod.cpu.repack.nibble4.ggml_gemm_mxfp4_8x8_q8_0_generic, .{ .name = "ggml_gemm_mxfp4_8x8_q8_0" });
        @export(&mod.cpu.repack.q2_k.ggml_gemm_q2_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q2_K_8x8_q8_K" });
    }
    // cpu/repack/x86/fallback.zig
    if (builtin.cpu.arch == .x86_64) {
        @export(&mod.cpu.repack.quantize.ggml_quantize_mat_q8_0_4x4_generic, .{ .name = "ggml_quantize_mat_q8_0_4x4" });
        @export(&mod.cpu.repack.quantize.ggml_quantize_mat_q8_K_4x4_generic, .{ .name = "ggml_quantize_mat_q8_K_4x4" });

        @export(&mod.cpu.repack.q1_0.ggml_gemv_q1_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q1_0_4x4_q8_0" });
        @export(&mod.cpu.repack.q1_0.ggml_gemv_q1_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q1_0_4x8_q8_0" });
        @export(&mod.cpu.repack.q4_0.ggml_gemv_q4_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q4_0_4x4_q8_0" });
        @export(&mod.cpu.repack.q4_0.ggml_gemv_q4_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q4_0_4x8_q8_0" });
        @export(&mod.cpu.repack.q4_k.ggml_gemv_q4_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q4_K_8x4_q8_K" });
        @export(&mod.cpu.repack.q5_k.ggml_gemv_q5_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q5_K_8x4_q8_K" });
        @export(&mod.cpu.repack.q5_k.ggml_gemv_q5_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q5_K_8x8_q8_K" });
        @export(&mod.cpu.repack.q6_k.ggml_gemv_q6_K_8x4_q8_K_generic, .{ .name = "ggml_gemv_q6_K_8x4_q8_K" });
        @export(&mod.cpu.repack.q6_k.ggml_gemv_q6_K_8x8_q8_K_generic, .{ .name = "ggml_gemv_q6_K_8x8_q8_K" });
        @export(&mod.cpu.repack.nibble4.ggml_gemv_iq4_nl_4x4_q8_0_generic, .{ .name = "ggml_gemv_iq4_nl_4x4_q8_0" });
        @export(&mod.cpu.repack.nibble4.ggml_gemv_mxfp4_4x4_q8_0_generic, .{ .name = "ggml_gemv_mxfp4_4x4_q8_0" });
        @export(&mod.cpu.repack.q8_0.ggml_gemv_q8_0_4x4_q8_0_generic, .{ .name = "ggml_gemv_q8_0_4x4_q8_0" });
        @export(&mod.cpu.repack.q8_0.ggml_gemv_q8_0_4x8_q8_0_generic, .{ .name = "ggml_gemv_q8_0_4x8_q8_0" });

        @export(&mod.cpu.repack.q1_0.ggml_gemm_q1_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q1_0_4x4_q8_0" });
        @export(&mod.cpu.repack.q1_0.ggml_gemm_q1_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q1_0_4x8_q8_0" });
        @export(&mod.cpu.repack.q4_0.ggml_gemm_q4_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q4_0_4x4_q8_0" });
        @export(&mod.cpu.repack.q4_0.ggml_gemm_q4_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q4_0_4x8_q8_0" });
        @export(&mod.cpu.repack.q4_k.ggml_gemm_q4_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q4_K_8x4_q8_K" });
        @export(&mod.cpu.repack.q5_k.ggml_gemm_q5_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q5_K_8x4_q8_K" });
        @export(&mod.cpu.repack.q5_k.ggml_gemm_q5_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q5_K_8x8_q8_K" });
        @export(&mod.cpu.repack.q6_k.ggml_gemm_q6_K_8x4_q8_K_generic, .{ .name = "ggml_gemm_q6_K_8x4_q8_K" });
        @export(&mod.cpu.repack.q6_k.ggml_gemm_q6_K_8x8_q8_K_generic, .{ .name = "ggml_gemm_q6_K_8x8_q8_K" });
        @export(&mod.cpu.repack.nibble4.ggml_gemm_iq4_nl_4x4_q8_0_generic, .{ .name = "ggml_gemm_iq4_nl_4x4_q8_0" });
        @export(&mod.cpu.repack.nibble4.ggml_gemm_mxfp4_4x4_q8_0_generic, .{ .name = "ggml_gemm_mxfp4_4x4_q8_0" });
        @export(&mod.cpu.repack.q8_0.ggml_gemm_q8_0_4x4_q8_0_generic, .{ .name = "ggml_gemm_q8_0_4x4_q8_0" });
        @export(&mod.cpu.repack.q8_0.ggml_gemm_q8_0_4x8_q8_0_generic, .{ .name = "ggml_gemm_q8_0_4x8_q8_0" });
    }
    // cpu/unary_ops.zig
    for (@typeInfo(mod.cpu.unary_ops.Op).@"enum".field_names) |name| {
        @export(mod.cpu.unary_ops.entry(@field(mod.cpu.unary_ops.Op, name)), .{ .name = "ggml_compute_forward_" ++ name, .linkage = .strong });
    }
    // cuda/backend.zig
    if (config.cuda_port) {
        @export(&mod.cuda.backend.init, .{ .name = "ggml_backend_cuda_init" });
        @export(&mod.cuda.backend.isCuda, .{ .name = "ggml_backend_is_cuda" });
        @export(&mod.cuda.backend.bufferType, .{ .name = "ggml_backend_cuda_buffer_type" });
        @export(&mod.cuda.backend.hostBufferType, .{ .name = "ggml_backend_cuda_host_buffer_type" });
        @export(&mod.cuda.backend.getDeviceCount, .{ .name = "ggml_backend_cuda_get_device_count" });
        @export(&mod.cuda.backend.getDeviceDescription, .{ .name = "ggml_backend_cuda_get_device_description" });
        @export(&mod.cuda.backend.getDeviceMemory, .{ .name = "ggml_backend_cuda_get_device_memory" });
        @export(&mod.cuda.backend.registerHostBuffer, .{ .name = "ggml_backend_cuda_register_host_buffer" });
        @export(&mod.cuda.backend.unregisterHostBuffer, .{ .name = "ggml_backend_cuda_unregister_host_buffer" });
        @export(&mod.cuda.backend.reg, .{ .name = "ggml_backend_cuda_reg" });
    }
    // metal/tuning.zig
    if (config.use_metal) {
        @export(&mod.metal.tuning.abiNe11Bucket, .{ .name = "_ZN17ggml_metal_tuning18fa_vec_ne11_bucketEx" });
        @export(&mod.metal.tuning.abiNe01Bucket, .{ .name = "_ZN17ggml_metal_tuning18fa_vec_ne01_bucketEx" });
        @export(&mod.metal.tuning.abiBaselineNe, .{ .name = "_ZN17ggml_metal_tuning18fa_vec_baseline_neEii" });
        @export(&mod.metal.tuning.abiBaselineCfg, .{ .name = "_ZN17ggml_metal_tuning19fa_vec_baseline_cfgEii" });
        @export(&mod.metal.tuning.abiSetOverride, .{ .name = "_ZN17ggml_metal_tuning19fa_vec_set_overrideENS_12fa_vec_cfg_tE" });
        @export(&mod.metal.tuning.abiClearOverride, .{ .name = "_ZN17ggml_metal_tuning21fa_vec_clear_overrideEv" });
        @export(&mod.metal.tuning.abiPick, .{ .name = "_ZN17ggml_metal_tuning11fa_vec_pickEiiiixx" });
    }
    // ops.zig
    for (mod.ops.activations) |entry| {
        @export(&mod.ops.activationWrapper(entry[1], entry[2]), .{ .name = entry[0] });
    }
    // ops.zig
    for (mod.ops.glus) |entry| {
        @export(&mod.ops.gluWrapper(entry[1], entry[2]), .{ .name = entry[0] });
    }
    for (mod.ops.glu_splits) |entry| {
        @export(&mod.ops.gluSplitWrapper(entry[1]), .{ .name = entry[0] });
    }
    // ops.zig
    for (mod.ops.norms) |entry| {
        @export(&mod.ops.normWrapper(entry[1], entry[2]), .{ .name = entry[0] });
    }
    // opt.zig
    for (mod.opt.exported_names) |name| @export(&@field(mod.opt, name), .{ .name = name });
    // quants/chunks.zig
    @export(&mod.quants.chunks.forwardingChunk(mod.quants.blocks.Q8_0, mod.impl.ggml_c.QK8_0, mod.quants.legacy.quantize_row_q8_0_ref), .{ .name = "quantize_q8_0" });
    @export(&mod.quants.chunks.forwardingChunk(mod.quants.blocks.MXFP4, mod.impl.ggml_c.QK_MXFP4, mod.quants.legacy.quantize_row_mxfp4_ref), .{ .name = "quantize_mxfp4" });
    @export(&mod.quants.chunks.forwardingChunk(mod.quants.blocks.NVFP4, mod.impl.ggml_c.QK_NVFP4, mod.quants.legacy.quantize_row_nvfp4_ref), .{ .name = "quantize_nvfp4" });
    @export(&mod.quants.chunks.forwardingChunk(mod.impl.ggml_c.ggml_fp8_e4m3_t, 1, mod.quants.legacy.quantize_row_f8_e4m3_ref), .{ .name = "quantize_f8_e4m3" });
    @export(&mod.quants.chunks.forwardingChunk(mod.quants.blocks.TQ1_0, mod.impl.ggml_c.QK_K, mod.quants.ternary.quantize_row_tq1_0_ref), .{ .name = "quantize_tq1_0" });
    @export(&mod.quants.chunks.forwardingChunk(mod.quants.blocks.TQ2_0, mod.impl.ggml_c.QK_K, mod.quants.ternary.quantize_row_tq2_0_ref), .{ .name = "quantize_tq2_0" });
    // quants/chunks.zig
    @export(&mod.quants.chunks.implChunk(mod.quants.blocks.Q4_0, mod.impl.ggml_c.QK4_0, mod.quants.legacy.quantize_row_q4_0_ref, mod.quants.chunks.q4_0Impl), .{ .name = "quantize_q4_0" });
    @export(&mod.quants.chunks.implChunk(mod.quants.blocks.Q4_1, mod.impl.ggml_c.QK4_1, mod.quants.legacy.quantize_row_q4_1_ref, mod.quants.chunks.q4_1Impl), .{ .name = "quantize_q4_1" });
    @export(&mod.quants.chunks.implChunk(mod.quants.blocks.Q5_0, mod.impl.ggml_c.QK5_0, mod.quants.legacy.quantize_row_q5_0_ref, mod.quants.chunks.q5_0Impl), .{ .name = "quantize_q5_0" });
    @export(&mod.quants.chunks.implChunk(mod.quants.blocks.Q5_1, mod.impl.ggml_c.QK5_1, mod.quants.legacy.quantize_row_q5_1_ref, mod.quants.chunks.q5_1Impl), .{ .name = "quantize_q5_1" });
    // quants/chunks.zig
    @export(&mod.quants.chunks.implChunk(mod.quants.blocks.Q2_K, mod.impl.ggml_c.QK_K, mod.quants.k.quantize_row_q2_K_ref, mod.quants.chunks.q2_KImpl), .{ .name = "quantize_q2_K" });
    @export(&mod.quants.chunks.implChunk(mod.quants.blocks.Q3_K, mod.impl.ggml_c.QK_K, mod.quants.k.quantize_row_q3_K_ref, mod.quants.chunks.q3_KImpl), .{ .name = "quantize_q3_K" });
    @export(&mod.quants.chunks.implChunk(mod.quants.blocks.Q4_K, mod.impl.ggml_c.QK_K, mod.quants.k.quantize_row_q4_K_ref, mod.quants.chunks.q4_KImpl), .{ .name = "quantize_q4_K" });
    @export(&mod.quants.chunks.implChunk(mod.quants.blocks.Q5_K, mod.impl.ggml_c.QK_K, mod.quants.k.quantize_row_q5_K_ref, mod.quants.chunks.q5_KImpl), .{ .name = "quantize_q5_K" });
    @export(&mod.quants.chunks.implChunk(mod.quants.blocks.Q6_K, mod.impl.ggml_c.QK_K, mod.quants.k.quantize_row_q6_K_ref, mod.quants.chunks.q6_KImpl), .{ .name = "quantize_q6_K" });
    // runtime.zig
    if (mod.impl.zig_varargs) {
        @export(&mod.runtime.ggmlAbort, .{ .name = "ggml_abort" });
        @export(&mod.runtime.ggmlLogInternal, .{ .name = "ggml_log_internal" });
    } else {
        @export(&mod.runtime.abortWith, .{ .name = "ggml_zig_abort_message" });
        @export(&mod.runtime.logEmit, .{ .name = "ggml_zig_log_emit" });
    }
    // vulkan/backend.zig
    if (config.vulkan_port) {
        @export(&mod.vulkan.backend.init, .{ .name = "ggml_backend_vk_init" });
        @export(&mod.vulkan.backend.isVk, .{ .name = "ggml_backend_is_vk" });
        @export(&mod.vulkan.backend.getDeviceCount, .{ .name = "ggml_backend_vk_get_device_count" });
        @export(&mod.vulkan.backend.getDeviceDescription, .{ .name = "ggml_backend_vk_get_device_description" });
        @export(&mod.vulkan.backend.getDeviceMemory, .{ .name = "ggml_backend_vk_get_device_memory" });
        @export(&mod.vulkan.buffer.bufferType, .{ .name = "ggml_backend_vk_buffer_type" });
        @export(&mod.vulkan.buffer.hostBufferType, .{ .name = "ggml_backend_vk_host_buffer_type" });
        @export(&mod.vulkan.backend.reg, .{ .name = "ggml_backend_vk_reg" });
        @export(&mod.vulkan.backend.conv_block_sizes, .{ .name = "vk_conv_block_sizes", .linkage = .weak });
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
