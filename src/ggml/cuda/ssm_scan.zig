//! `GGML_OP_SSM_SCAN`: the selective scan of Mamba-1 and Mamba-2, and for
//! long Mamba-2 prefills the chunked State Space Duality path, which turns
//! the scan into cuBLAS matmuls.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cuda/ssm-scan.cu` and
//! `llama.cpp/ggml/src/ggml-cuda/ssm-scan.cuh` at v0.3.0 (`c1d0e7a00`). The
//! kernels -- `ssm_scan_f32`, `ssm_scan_f32_group` and the four
//! `ssm_ssd_*` kernels -- stay CUDA. The SSD arm is the C's
//! `!defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)` one, which this
//! build compiles.
//!
//! # The cuBLAS handle's stream
//!
//! The SSD path calls cuBLAS through `ctx.cublas_handle()` **without** a
//! `cublasSetStream` of its own, unlike `ggml_cuda_mul_mat_batched_cublas`.
//! The handle is per (device, stream) and bound to its stream when it is
//! created, so the result is the same; the port does not add the call.

const std = @import("std");
const impl = @import("../impl.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");

const c = impl.c;
const cu = common.cu;
const Tensor = c.ggml_tensor;
const half = kernels.half;

/// Ports `SSM_SSD_MIN_TOKENS` (ssm-scan.cu:15 @c1d0e7a00).
const ssd_min_tokens = 128;
/// Ports `SSM_SSD_DT_BLOCK` (ssm-scan.cu:18 @c1d0e7a00).
const ssd_dt_block = 256;
/// Ports `SSM_SSD_DT_MAX_ITEMS` (ssm-scan.cu:19 @c1d0e7a00).
const ssd_dt_max_items = 32;
/// Ports `SSM_SSD_MAX_TOKENS` (ssm-scan.cu:22 @c1d0e7a00).
const ssd_max_tokens = ssd_dt_block * ssd_dt_max_items;
/// Ports `SSM_SSD_CHUNK_SIZE` (ssm-scan.cu:25 @c1d0e7a00).
const ssd_chunk_size = 256;

const int_max: usize = std.math.maxInt(c_int);

/// A `size_t` or `int64_t` passed into an `int` parameter: C++ converts
/// modulo 2^32.
fn int(x: anytype) c_int {
    return switch (@TypeOf(x)) {
        usize, u64 => @bitCast(@as(u32, @truncate(x))),
        i64 => @truncate(x),
        c_int => x,
        else => @compileError("int: " ++ @typeName(@TypeOf(x))),
    };
}

/// An `int64_t` into a `dim3` member, which is `unsigned int`.
fn dim(x: i64) u32 {
    return @truncate(@as(u64, @bitCast(x)));
}

/// `p + off` on a typed device pointer, `off` in elements.
fn at(p: anytype, off: i64) @TypeOf(p) {
    return if (off >= 0) p + @as(usize, @intCast(off)) else p - @as(usize, @intCast(-off));
}

/// Ports `ssm_scan_f32_cuda` (ssm-scan.cu:239 @c1d0e7a00).
fn ssmScanF32Cuda(
    src0: ?*const anyopaque,
    src1: ?*const anyopaque,
    src2: ?*const anyopaque,
    src3: ?*const anyopaque,
    src4: ?*const anyopaque,
    src5: ?*const anyopaque,
    src6: ?*const anyopaque,
    dst: ?*anyopaque,
    src0_nb2: c_int,
    src0_nb3: c_int,
    src1_nb2: c_int,
    src1_nb3: c_int,
    src2_nb1: c_int,
    src2_nb2: c_int,
    src3_nb1: c_int,
    src4_nb2: c_int,
    src4_nb3: c_int,
    src5_nb2: c_int,
    src5_nb3: c_int,
    s_off: i64,
    d_state: i64,
    head_dim: i64,
    n_head: i64,
    n_group: i64,
    n_tok: i64,
    n_seq: i64,
    K: i64,
    stream: cu.cudaStream_t,
) void {
    const ptrs = .{ src0, src1, src2, src3, src4, src5, src6, dst };
    const nbs = .{ src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3 };
    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    if (src3_nb1 == @sizeOf(f32)) {
        // Mamba-2
        if (d_state == 128) {
            const threads = 128;
            const num_warps = threads / common.warp_size;

            const blocks: kernels.Dim3 = .{ .x = dim(@divTrunc(n_head * head_dim + (num_warps - 1), num_warps)), .y = dim(n_seq), .z = 1 };
            kernels.launchPdl(kernels.get("ssm-scan", kernels.tmpl("ssm_scan_f32_group", &.{ kernels.int(128 / common.warp_size), kernels.int(128) })), .{
                .grid = blocks,
                .block = .{ .x = threads },
                .shmem = 0,
                .stream = stream,
            }, ptrs ++ nbs ++ .{ s_off, n_head, head_dim, n_group, n_tok, K });
        } else if (d_state == 256) { // Falcon-H1
            const threads = 256;
            const num_warps = threads / common.warp_size;

            const blocks: kernels.Dim3 = .{ .x = dim(@divTrunc(n_head * head_dim + (num_warps - 1), num_warps)), .y = dim(n_seq), .z = 1 };
            kernels.launchPdl(kernels.get("ssm-scan", kernels.tmpl("ssm_scan_f32_group", &.{ kernels.int(256 / common.warp_size), kernels.int(256) })), .{
                .grid = blocks,
                .block = .{ .x = threads },
                .shmem = 0,
                .stream = stream,
            }, ptrs ++ nbs ++ .{ s_off, n_head, head_dim, n_group, n_tok, K });
        } else {
            impl.abort("doesn't support d_state!=(128 or 256).");
        }
    } else {
        // Mamba-1
        impl.assert(K == 1, "K == 1");
        const threads = 128;
        impl.assert(@rem(n_head, threads) == 0, "n_head % threads == 0");
        impl.assert(head_dim == 1, "head_dim == 1");
        impl.assert(n_group == 1, "n_group == 1");
        const blocks: kernels.Dim3 = .{ .x = dim(n_seq), .y = dim(@divTrunc(n_head + threads - 1, threads)), .z = 1 };
        if (d_state == 16) {
            const params: kernels.LaunchParams = .{ .grid = blocks, .block = .{ .x = threads }, .shmem = 0, .stream = stream };
            const args = ptrs ++ nbs ++ .{ s_off, n_head, n_tok };
            switch (n_tok) {
                inline 1...8 => |l| kernels.launchPdl(scanKernel(threads, l), params, args),
                else => kernels.launchPdl(scanKernel(threads, 0), params, args),
            }
        } else {
            impl.abort("doesn't support d_state!=16.");
        }
    }
}

fn scanKernel(comptime threads: comptime_int, comptime l: comptime_int) kernels.Kernel {
    return kernels.get("ssm-scan", kernels.tmpl("ssm_scan_f32", &.{ kernels.ulong(threads), kernels.ulong(16), kernels.ulong(l) }));
}

/// Ports `ssm_scan_ssd_f32_cuda` (ssm-scan.cu:576 @c1d0e7a00): the SSD
/// dispatch for Mamba-2 prefill. All strides are in elements, not bytes.
fn ssmScanSsdF32Cuda(
    ctx: *common.Context,
    src0_d: ?*const anyopaque,
    src1_d: ?*const anyopaque,
    src2_d: ?*const anyopaque,
    src3_d: ?*const anyopaque,
    src4_d: [*]const f32,
    src5_d: [*]const f32,
    src6_d: ?*const anyopaque,
    dst_d: [*]f32,
    s0_stride_seq: i64,
    x_stride_tok: c_int,
    x_stride_seq: c_int,
    dt_stride_tok: c_int,
    dt_stride_seq: c_int,
    A_stride: c_int,
    B_stride_tok: c_int,
    B_stride_seq: c_int,
    C_stride_tok: c_int,
    C_stride_seq: c_int,
    s_off: i64,
    d_state: i64,
    head_dim: i64,
    n_head: i64,
    n_group: i64,
    n_tok: i64,
    n_seq: i64,
) void {
    const stream = ctx.stream();
    const d_inner = head_dim * n_head;

    const chunk_size: i64 = ssd_chunk_size;
    const n_chunks = @divTrunc(n_tok + chunk_size - 1, chunk_size);

    const state_per_head = d_state * head_dim;

    const matmul_dtype = cu.CUDA_R_16F;

    const sz = struct {
        fn f(x: i64) usize {
            return @bitCast(x);
        }
    }.f;

    var dt_sp_buf = common.PoolAlloc(f32).initAlloc(ctx.pool(), sz(n_tok * n_head * n_seq));
    defer dt_sp_buf.deinit();
    var cs_buf = common.PoolAlloc(f32).initAlloc(ctx.pool(), sz(n_tok * n_head * n_seq));
    defer cs_buf.deinit();
    var CB_buf = common.PoolAlloc(f32).initAlloc(ctx.pool(), sz(chunk_size * chunk_size * n_group * n_seq));
    defer CB_buf.deinit();
    var X_dt_buf = common.PoolAlloc(half).initAlloc(ctx.pool(), sz(chunk_size * head_dim * n_head * n_seq));
    defer X_dt_buf.deinit();
    var B_w_buf = common.PoolAlloc(half).initAlloc(ctx.pool(), sz(d_state * chunk_size * n_head * n_seq));
    defer B_w_buf.deinit();
    var C_s_buf = common.PoolAlloc(f32).initAlloc(ctx.pool(), sz(d_state * chunk_size * n_head * n_seq));
    defer C_s_buf.deinit();
    const dt_sp = dt_sp_buf.get().?;
    const cs = cs_buf.get().?;
    const CB = CB_buf.get().?;
    const X_dt = X_dt_buf.get().?;
    const B_weighted = B_w_buf.get().?;
    const C_scaled = C_s_buf.get().?;
    // write state directly to dst
    const s_cur: [*]f32 = @ptrCast(@alignCast(@as([*]u8, @ptrCast(dst_d)) + @as(usize, @intCast(s_off))));

    // Step 1: softplus(dt) and parallel prefix sum over full sequence
    {
        const grid: kernels.Dim3 = .{ .x = dim(n_head), .y = dim(n_seq) };
        kernels.launch(kernels.get("ssm-scan", kernels.tmpl("ssm_ssd_prepare_dt_kernel", &.{ kernels.int(ssd_dt_block), kernels.int(ssd_dt_max_items) })), .{
            .grid = grid,
            .block = .{ .x = ssd_dt_block },
            .shmem = 0,
            .stream = stream,
        }, .{ src2_d, @as(?*anyopaque, dt_sp), @as(?*anyopaque, cs), int(n_head), int(n_tok), dt_stride_tok, dt_stride_seq });
        common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());
    }

    // Step 2: initialize running state from src0[ids[s]]
    {
        const BLOCK = 256;
        const state_size = d_state * head_dim * n_head;
        const grid: kernels.Dim3 = .{ .x = dim(@divTrunc(state_size + BLOCK - 1, BLOCK)), .y = dim(n_seq) };
        kernels.launch(kernels.get("ssm-scan", kernels.tmpl("ssm_ssd_init_state_kernel", &.{kernels.int(BLOCK)})), .{
            .grid = grid,
            .block = .{ .x = BLOCK },
            .shmem = 0,
            .stream = stream,
        }, .{ src0_d, src6_d, @as(?*anyopaque, s_cur), int(state_size), s0_stride_seq });
        common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());
    }

    // Step 3: chunked SSD loop
    // Per chunk: pre_matmul (incl. M) + 4 cuBLAS (CB, Y, S@C, state update) + scale_state
    const handle = ctx.cublasHandle();
    const alpha_one: f32 = 1.0;
    const beta_zero: f32 = 0.0;
    const beta_one: f32 = 1.0;
    const lda_C_src = C_stride_tok; // leading dim for C in CB = C^T @ B
    const ldb_B_src = B_stride_tok; // leading dim for B in CB = C^T @ B

    // Scratch buffer for causal M matrix, reused across chunks (max size at chunk_size)
    const n_M_max = chunk_size * chunk_size;
    var M_buf = common.PoolAlloc(half).initAlloc(ctx.pool(), sz(n_M_max * n_head * n_seq));
    defer M_buf.deinit();
    const M_mat = M_buf.get().?;

    var k: i64 = 0;
    while (k < n_chunks) : (k += 1) {
        const chunk_offset = k * chunk_size;
        const chunk_len = if (chunk_offset + chunk_size <= n_tok) chunk_size else (n_tok - chunk_offset);

        // 3a: CB = C^T @ B per group
        {
            var s: i64 = 0;
            while (s < n_seq) : (s += 1) {
                const C_s = at(src5_d, s * C_stride_seq + chunk_offset * C_stride_tok);
                const B_s = at(src4_d, s * B_stride_seq + chunk_offset * B_stride_tok);
                const CB_s = at(CB, s * chunk_len * chunk_len * n_group);

                if (n_group == 1) {
                    common.cublasCheck(cu.cublasSgemm_v2(handle, cu.CUBLAS_OP_T, cu.CUBLAS_OP_N, int(chunk_len), int(chunk_len), int(d_state), &alpha_one, C_s, lda_C_src, B_s, ldb_B_src, &beta_zero, CB_s, int(chunk_len)), "cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, chunk_len, chunk_len, d_state, &alpha_one, C_s, lda_C_src, B_s, ldb_B_src, &beta_zero, CB_s, (int)chunk_len)", @src());
                } else {
                    common.cublasCheck(cu.cublasGemmStridedBatchedEx(handle, cu.CUBLAS_OP_T, cu.CUBLAS_OP_N, int(chunk_len), int(chunk_len), int(d_state), &alpha_one, C_s, cu.CUDA_R_32F, lda_C_src, d_state, B_s, cu.CUDA_R_32F, ldb_B_src, d_state, &beta_zero, CB_s, cu.CUDA_R_32F, int(chunk_len), chunk_len * chunk_len, int(n_group), cu.CUBLAS_COMPUTE_32F, cu.CUBLAS_GEMM_DEFAULT), "cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, chunk_len, chunk_len, d_state, &alpha_one, C_s, CUDA_R_32F, lda_C_src, d_state, B_s, CUDA_R_32F, ldb_B_src, d_state, &beta_zero, CB_s, CUDA_R_32F, (int)chunk_len, (long long)(chunk_len * chunk_len), n_group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)", @src());
                }
            }
        }

        // 3b: prepare X_dt, B_weighted, C_scaled + materialize causal M matrix
        const n_M = chunk_len * chunk_len;
        {
            const BLOCK = 256;
            const n_xdt = chunk_len * head_dim;
            const n_bw = d_state * chunk_len;
            var max_work = n_xdt;
            if (n_bw > max_work) max_work = n_bw;
            if (n_M > max_work) max_work = n_M;
            const grid: kernels.Dim3 = .{ .x = dim(@divTrunc(max_work + BLOCK - 1, BLOCK)), .y = dim(n_head), .z = dim(n_seq) };
            kernels.launch(kernels.get("ssm-scan", kernels.tmpl("ssm_ssd_pre_matmul_kernel", &.{ kernels.int(BLOCK), kernels.typeName(half) })), .{
                .grid = grid,
                .block = .{ .x = BLOCK },
                .shmem = 0,
                .stream = stream,
            }, .{
                @as(?*const anyopaque, cs), @as(?*const anyopaque, dt_sp),  src3_d,
                src1_d,                     @as(?*const anyopaque, src4_d), @as(?*const anyopaque, src5_d),
                @as(?*anyopaque, X_dt),     @as(?*anyopaque, B_weighted),   @as(?*anyopaque, C_scaled),
                @as(?*const anyopaque, CB), @as(?*anyopaque, M_mat),        int(chunk_len),
                int(head_dim),              int(n_head),                    int(n_group),
                int(d_state),               A_stride,                       x_stride_tok,
                x_stride_seq,               B_stride_tok,                   B_stride_seq,
                C_stride_tok,               C_stride_seq,                   int(chunk_offset),
                int(n_tok),
            });
            common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());
        }

        // 3c: dst = S_cur^T @ C_scaled (state contribution)
        {
            const stride_S = state_per_head;
            const stride_Cs = d_state * chunk_len;

            var s: i64 = 0;
            while (s < n_seq) : (s += 1) {
                const dst_chunk = at(dst_d, s * d_inner * n_tok + chunk_offset * d_inner);

                common.cublasCheck(cu.cublasGemmStridedBatchedEx(handle, cu.CUBLAS_OP_T, cu.CUBLAS_OP_N, int(head_dim), int(chunk_len), int(d_state), &alpha_one, at(s_cur, s * stride_S * n_head), cu.CUDA_R_32F, int(d_state), stride_S, at(C_scaled, s * stride_Cs * n_head), cu.CUDA_R_32F, int(d_state), stride_Cs, &beta_zero, dst_chunk, cu.CUDA_R_32F, int(d_inner), head_dim, int(n_head), cu.CUBLAS_COMPUTE_32F, cu.CUBLAS_GEMM_DEFAULT), "cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, head_dim, chunk_len, d_state, &alpha_one, s_cur + s * stride_S * n_head, CUDA_R_32F, d_state, stride_S, C_scaled + s * stride_Cs * n_head, CUDA_R_32F, d_state, stride_Cs, &beta_zero, dst_chunk, CUDA_R_32F, d_inner, head_dim, n_head, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)", @src());
            }
        }

        // 3d: dst += X_dt @ M^T (intra-chunk contribution, adds to 3c result)
        // M is stored as M[t_out, t_in] (lower-triangular), transpose needed for Y = X @ M^T.
        {
            const stride_M = n_M;
            const stride_X_h = chunk_len * head_dim;

            var s: i64 = 0;
            while (s < n_seq) : (s += 1) {
                const dst_chunk = at(dst_d, s * d_inner * n_tok + chunk_offset * d_inner);
                common.cublasCheck(cu.cublasGemmStridedBatchedEx(handle, cu.CUBLAS_OP_N, cu.CUBLAS_OP_T, int(head_dim), int(chunk_len), int(chunk_len), &alpha_one, at(X_dt, s * stride_X_h * n_head), matmul_dtype, int(head_dim), stride_X_h, at(M_mat, s * stride_M * n_head), matmul_dtype, int(chunk_len), stride_M, &beta_one, dst_chunk, cu.CUDA_R_32F, int(d_inner), head_dim, int(n_head), cu.CUBLAS_COMPUTE_32F, cu.CUBLAS_GEMM_DEFAULT), "cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, head_dim, chunk_len, chunk_len, &alpha_one, X_dt + s * stride_X_h * n_head, matmul_dtype, head_dim, stride_X_h, M_mat + s * stride_M * n_head, matmul_dtype, chunk_len, stride_M, &beta_one, dst_chunk, CUDA_R_32F, d_inner, head_dim, n_head, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)", @src());
            }
        }

        // 3e: s_cur = B_weighted @ X_dt^T + decay_total * s_cur_old (state update)
        {
            // Scale s_cur in-place by per-head decay_total BEFORE cuBLAS overwrites it
            const BLOCK = 256;
            const grid: kernels.Dim3 = .{ .x = dim(@divTrunc(state_per_head + BLOCK - 1, BLOCK)), .y = dim(n_head), .z = dim(n_seq) };
            kernels.launch(kernels.get("ssm-scan", kernels.tmpl("ssm_ssd_scale_state_kernel", &.{kernels.int(BLOCK)})), .{
                .grid = grid,
                .block = .{ .x = BLOCK },
                .shmem = 0,
                .stream = stream,
            }, .{
                @as(?*anyopaque, s_cur), @as(?*const anyopaque, cs), src3_d,
                int(d_state),            int(head_dim),              int(n_head),
                int(chunk_offset),       int(chunk_len),             int(n_tok),
                A_stride,
            });
            common.cudaCheck(cu.cudaGetLastError(), "cudaGetLastError()", @src());

            // cuBLAS with beta=1: s_cur = B_weighted @ X_dt^T + 1.0 * s_cur (already scaled)
            const stride_Bw = d_state * chunk_len;
            const stride_X = chunk_len * head_dim;
            const stride_S = state_per_head;

            var s: i64 = 0;
            while (s < n_seq) : (s += 1) {
                // X_dt is d-fastest {hd, C}, read as OP_T to get {C, hd}
                common.cublasCheck(cu.cublasGemmStridedBatchedEx(handle, cu.CUBLAS_OP_N, cu.CUBLAS_OP_T, int(d_state), int(head_dim), int(chunk_len), &alpha_one, at(B_weighted, s * stride_Bw * n_head), matmul_dtype, int(d_state), stride_Bw, at(X_dt, s * stride_X * n_head), matmul_dtype, int(head_dim), stride_X, &beta_one, at(s_cur, s * stride_S * n_head), cu.CUDA_R_32F, int(d_state), stride_S, int(n_head), cu.CUBLAS_COMPUTE_32F, cu.CUBLAS_GEMM_DEFAULT), "cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, d_state, head_dim, chunk_len, &alpha_one, B_weighted + s * stride_Bw * n_head, matmul_dtype, d_state, stride_Bw, X_dt + s * stride_X * n_head, matmul_dtype, head_dim, stride_X, &beta_one, s_cur + s * stride_S * n_head, CUDA_R_32F, d_state, stride_S, n_head, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)", @src());
            }
        }
    }
}

/// Ports `ggml_cuda_op_ssm_scan` (ssm-scan.cu:767 @c1d0e7a00).
pub fn opSsmScan(ctx: *common.Context, dst: *Tensor) void {
    const src0 = impl.one(Tensor, dst.src[0]); // s
    const src1 = impl.one(Tensor, dst.src[1]); // x
    const src2 = impl.one(Tensor, dst.src[2]); // dt
    const src3 = impl.one(Tensor, dst.src[3]); // A
    const src4 = impl.one(Tensor, dst.src[4]); // B
    const src5 = impl.one(Tensor, dst.src[5]); // C
    const src6 = impl.one(Tensor, dst.src[6]); // ids

    const nc = src0.ne[0]; // d_state
    const nr = src0.ne[1]; // head_dim or 1
    const nh = src1.ne[1]; // n_head
    const ng = src4.ne[1]; // n_group
    const n_t = src1.ne[2]; // number of tokens per sequence
    const n_s = src1.ne[3]; // number of sequences in the batch
    const K_param = impl.getOpParamsI32(dst, 0);
    const K: i64 = if (K_param > 0) K_param else 1;

    const s_off: i64 = @bitCast(@as(usize, @bitCast(c.ggml_nelements(src1))) * @sizeOf(f32));

    impl.assert(c.ggml_nelements(src1) + K * nc * nr * nh * n_s == c.ggml_nelements(dst), "ggml_nelements(src1) + K*nc*nr*nh*n_s == ggml_nelements(dst)");
    impl.assert(src0.nb[0] == @sizeOf(f32), "src0->nb[0] == sizeof(float)");
    impl.assert(src1.nb[0] == @sizeOf(f32), "src1->nb[0] == sizeof(float)");
    impl.assert(src2.nb[0] == @sizeOf(f32), "src2->nb[0] == sizeof(float)");
    impl.assert(src3.nb[0] == @sizeOf(f32), "src3->nb[0] == sizeof(float)");
    impl.assert(src4.nb[0] == @sizeOf(f32), "src4->nb[0] == sizeof(float)");
    impl.assert(src5.nb[0] == @sizeOf(f32), "src5->nb[0] == sizeof(float)");
    impl.assert(src6.nb[0] == @sizeOf(i32), "src6->nb[0] == sizeof(int32_t)");
    impl.assert(src3.ne[0] == 1 or K == 1, "src3->ne[0] == 1 || K == 1");

    const src0_d: ?*const anyopaque = src0.data;
    const src1_d: ?*const anyopaque = src1.data;
    const src2_d: ?*const anyopaque = src2.data;
    const src3_d: ?*const anyopaque = src3.data;
    const src4_d: ?*const anyopaque = src4.data;
    const src5_d: ?*const anyopaque = src5.data;
    const src6_d: ?*const anyopaque = src6.data;
    const dst_d: ?*anyopaque = dst.data;
    const stream = ctx.stream();

    impl.assert(src0.type == c.GGML_TYPE_F32, "src0->type == GGML_TYPE_F32");
    impl.assert(src6.type == c.GGML_TYPE_I32, "src6->type == GGML_TYPE_I32");
    impl.assert(dst.type == c.GGML_TYPE_F32, "dst->type == GGML_TYPE_F32");

    // Byte strides are narrowed to int for both scan and SSD paths.
    impl.assert(src0.nb[2] <= int_max, "src0->nb[2] <= (size_t)INT_MAX");
    impl.assert(src0.nb[3] <= int_max, "src0->nb[3] <= (size_t)INT_MAX");
    impl.assert(src1.nb[2] <= int_max, "src1->nb[2] <= (size_t)INT_MAX");
    impl.assert(src1.nb[3] <= int_max, "src1->nb[3] <= (size_t)INT_MAX");
    impl.assert(src2.nb[1] <= int_max, "src2->nb[1] <= (size_t)INT_MAX");
    impl.assert(src2.nb[2] <= int_max, "src2->nb[2] <= (size_t)INT_MAX");
    impl.assert(src3.nb[1] <= int_max, "src3->nb[1] <= (size_t)INT_MAX");
    impl.assert(src4.nb[2] <= int_max, "src4->nb[2] <= (size_t)INT_MAX");
    impl.assert(src4.nb[3] <= int_max, "src4->nb[3] <= (size_t)INT_MAX");
    impl.assert(src5.nb[2] <= int_max, "src5->nb[2] <= (size_t)INT_MAX");
    impl.assert(src5.nb[3] <= int_max, "src5->nb[3] <= (size_t)INT_MAX");

    // Mamba-2 with scalar A per head: use SSD matmul path for long sequences.
    // Requires NVIDIA Turing+ otherwise fallback to scan.
    const is_mamba2 = src3.nb[1] == @sizeOf(f32);
    const cc = common.info().devices[@intCast(common.getDevice())].cc;
    const use_ssd = is_mamba2 and n_t > ssd_min_tokens and
        K == 1 and
        n_t <= ssd_max_tokens and
        common.cc.isNvidia(cc) and
        cc >= common.cc.turing and
        @rem(nr, 8) == 0; // cuBLAS requires 8-element (16-byte) alignment

    if (use_ssd) {
        // ssm_ssd_init_state_kernel uses flat linear indexing within each sequence,
        // so src0 must be fully contiguous across all inner dimensions.
        // The scan path handles non-contiguous nb[2] via src0_nb2 but does not handle nb[1].
        impl.assert(src0.nb[1] == @as(usize, @bitCast(nc)) * @sizeOf(f32), "src0->nb[1] == nc         * sizeof(float)");
        impl.assert(src0.nb[2] == @as(usize, @bitCast(nc * nr)) * @sizeOf(f32), "src0->nb[2] == nc * nr    * sizeof(float)");

        const f = @sizeOf(f32);
        ssmScanSsdF32Cuda(
            ctx,
            src0_d,
            src1_d,
            src2_d,
            src3_d,
            @ptrCast(@alignCast(src4_d)),
            @ptrCast(@alignCast(src5_d)),
            src6_d,
            @ptrCast(@alignCast(dst_d)),
            @intCast(src0.nb[3] / f),
            int(src1.nb[2] / f),
            int(src1.nb[3] / f),
            int(src2.nb[1] / f),
            int(src2.nb[2] / f),
            int(src3.nb[1] / f),
            int(src4.nb[2] / f),
            int(src4.nb[3] / f),
            int(src5.nb[2] / f),
            int(src5.nb[3] / f),
            s_off,
            nc,
            nr,
            nh,
            ng,
            n_t,
            n_s,
        );
        return;
    }
    ssmScanF32Cuda(src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d, int(src0.nb[2]), int(src0.nb[3]), int(src1.nb[2]), int(src1.nb[3]), int(src2.nb[1]), int(src2.nb[2]), int(src3.nb[1]), int(src4.nb[2]), int(src4.nb[3]), int(src5.nb[2]), int(src5.nb[3]), s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}
