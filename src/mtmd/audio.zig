//! Audio preprocessing: PCM samples to the log-mel spectrograms (or framed
//! waveforms) mtmd's audio encoders read, and a streaming inverse STFT for
//! the generators.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/tools/mtmd/mtmd-audio.cpp` and
//! `llama.cpp/tools/mtmd/mtmd-audio.h` at v1.0.0 (`b1b770d90`), and
//! `mtmd_audio_log_mel` of `llama.cpp/tools/mtmd/mtmd-subcontext.cpp`. Each
//! declaration names the C++ it replaces and the line it began at. The C++'s
//! class hierarchy -- an abstract `mtmd_audio_preprocessor` and eleven
//! subclasses -- is one `Preprocessor` here, switched on `Kind`.
//!
//! # Numerics
//!
//! `make mtmd-audio-diff` holds every preprocessor to the C++ on bits, so the
//! arithmetic is the C++'s as clang compiles it: `float` and `double` where it
//! wrote them, every libm call through the C library (Zig's `compiler_rt`
//! would otherwise shadow `expf` and `logf`), and every contraction clang
//! forms named with `@mulAdd`. Clang contracts a `+` or `-` whose left
//! operand is a multiply by fusing that one; otherwise a right-hand multiply,
//! negated under `-`. So `a*b + c*d` is `fma(a, b, c*d)`, the FFT butterfly
//! fuses both of its products in turn, and a `double` sum of a `float`
//! product never fuses: the types differ.
//!
//! The log-mel itself is a ggml graph (`logMel`, upstream's
//! `mtmd_audio_log_mel`): a windowed DFT as a `MUL_MAT`, the power, the
//! filterbank as a second `MUL_MAT`, the floor and the log, on whichever
//! backend the scheduler places it. Its bits are ggml's, so the graph is
//! built op for op as upstream builds it. `preprocess` gives the shapes with
//! no data and `compute` runs the graph later, on the encoding thread, as
//! upstream's `preprocess` and `compute` do; the C++ worker threads that
//! computed it on the CPU are gone.

const std = @import("std");
const builtin = @import("builtin");
const impl = @import("impl.zig");
const model = @import("model.zig");
const gpa = impl.gpa;
const c = impl.c;

const Hparams = model.Hparams;

const libm = struct {
    /// The `__float2` Apple's `__sincosf_stret` returns.
    const Float2 = extern struct { sin: f32, cos: f32 };
    extern "c" fn __sincosf_stret(x: f32) Float2;
    extern "c" fn __exp10(x: f64) f64;
    extern "c" fn sinf(x: f32) f32;
    extern "c" fn cosf(x: f32) f32;
    extern "c" fn log(x: f64) f64;
    extern "c" fn log10(x: f64) f64;
    extern "c" fn pow(x: f64, y: f64) f64;
    extern "c" fn exp(x: f64) f64;
    // chatterbox's reference front-end
    /// The `__double2` Apple's `__sincos_stret` returns.
    const Double2 = extern struct { sin: f64, cos: f64 };
    extern "c" fn __sincos_stret(x: f64) Double2;
    extern "c" fn __exp10f(x: f32) f32;
    extern "c" fn sin(x: f64) f64;
    extern "c" fn cos(x: f64) f64;
    extern "c" fn logf(x: f32) f32;
    extern "c" fn powf(x: f32, y: f32) f32;
};

/// Whether the reference compiler treats libm calls as writing `errno`:
/// clang's `-fmath-errno`, its default off Darwin. With it off -- Darwin --
/// clang folds a `sinf`/`cosf` pair on one argument into Apple's
/// `__sincosf_stret`, which rounds differently from the separate calls, and
/// turns `pow(10.0, x)` into `__exp10(x)`. Both measured from the imports of
/// the reference `mtmd-audio.o` (`___sincosf_stret`, `___exp10`); the same
/// rule as `src/ggml/cpu/simd.zig`'s `math_errno`.
const math_errno = !builtin.os.tag.isDarwin();

/// `sinf(x)` and `cosf(x)` as clang emits the pair.
pub fn sinCos(x: f32) [2]f32 {
    if (comptime !math_errno) {
        const r = libm.__sincosf_stret(x);
        return .{ r.sin, r.cos };
    }
    return .{ libm.sinf(x), libm.cosf(x) };
}

/// `pow(10.0, x)` as clang emits it.
fn pow10(x: f64) f64 {
    if (comptime !math_errno) return libm.__exp10(x);
    return libm.pow(10.0, x);
}

/// `M_PI`.
const m_pi: f64 = 3.14159265358979323846;

fn ggmlAssert(ok: bool, comptime line: c_int, comptime expr: [*:0]const u8) void {
    impl.ggmlAssert(ok, "mtmd-audio.cpp", line, expr);
}

/// `std::max(a, b)`: `(a < b) ? b : a`, which picks `a` on a NaN.
fn maxF64(a: f64, b: f64) f64 {
    return if (a < b) b else a;
}

/// Ports `mtmd_audio_mel` (mtmd-audio.h:13 @b1b770d90): `n_mel` rows of
/// `n_len` frames, row-major.
pub const Mel = struct {
    n_len: i64 = 0,
    n_len_org: i64 = 0,
    n_mel: i64 = 0,
    data: std.ArrayList(f32) = .empty,

    pub fn deinit(self: *Mel) void {
        self.data.deinit(gpa);
    }
};

/// Ports `mtmd_audio_mel_filters` (mtmd-audio.h:21 @b1b770d90).
pub const MelFilters = struct {
    n_mel: i64 = 0,
    n_fft: i64 = 0,
    data: std.ArrayList(f32) = .empty,
};

/// Ports `mtmd_audio_cache` (mtmd-audio.h:29 @b1b770d90): the sine and cosine
/// tables, the window and the filterbank one preprocessor owns.
pub const Cache = struct {
    sin_vals: std.ArrayList(f32) = .empty,
    cos_vals: std.ArrayList(f32) = .empty,
    hann_window: std.ArrayList(f32) = .empty,
    filters: MelFilters = .{},

    pub fn deinit(self: *Cache) void {
        self.sin_vals.deinit(gpa);
        self.cos_vals.deinit(gpa);
        self.hann_window.deinit(gpa);
        self.filters.data.deinit(gpa);
    }

    /// Ports `fill_sin_cos_table` (mtmd-audio.cpp:17 @b1b770d90), a member of `mtmd_audio_cache`.
    pub fn fillSinCosTable(self: *Cache, n: u32) !void {
        try self.sin_vals.resize(gpa, n);
        try self.cos_vals.resize(gpa, n);
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            const theta: f64 = (2 * m_pi * @as(f64, @floatFromInt(i))) / @as(f64, @floatFromInt(n));
            const sc = sinCos(@floatCast(theta));
            self.sin_vals.items[i] = sc[0];
            self.cos_vals.items[i] = sc[1];
        }
    }

    /// Ports `fill_hann_window` (mtmd-audio.cpp:27 @b1b770d90), a member of `mtmd_audio_cache`.
    /// `length + offset` is unsigned arithmetic in the C++.
    pub fn fillHannWindow(self: *Cache, length: u32, periodic: bool) !void {
        try self.hann_window.resize(gpa, length);
        const offset: c_int = if (periodic) 0 else -1;
        const denom: u32 = length +% @as(u32, @bitCast(offset));
        var i: u32 = 0;
        while (i < length) : (i += 1) {
            const arg: f64 = (2.0 * m_pi * @as(f64, @floatFromInt(i))) / @as(f64, @floatFromInt(denom));
            self.hann_window.items[i] = @floatCast(0.5 * (1.0 - @as(f64, libm.cosf(@floatCast(arg)))));
        }
    }

    /// Ports `fill_mel_filterbank_matrix` (mtmd-audio.cpp:35 @b1b770d90), a member of `mtmd_audio_cache`:
    /// an `n_mel` by `n_fft / 2 + 1` filterbank, Slaney or HTK scale.
    pub fn fillMelFilterbankMatrix(
        self: *Cache,
        n_mel: i64,
        n_fft: i64,
        sample_rate: c_int,
        fmin: f32,
        fmax_in: f32,
        slaney_area_norm: bool,
        scale: f32,
        use_htk: bool,
    ) !void {
        ggmlAssert(n_mel > 0 and n_fft > 1, 43, "n_mel > 0 && n_fft > 1");
        var fmax = fmax_in;
        if (fmax <= 0.0) fmax = 0.5 * @as(f32, @floatFromInt(sample_rate));

        // Slaney scale (matches librosa default)
        const min_log_hz: f64 = 1000.0;
        const lin_slope: f64 = 3.0 / 200.0;
        const min_log_mel: f64 = min_log_hz * lin_slope;
        const log_step: f64 = libm.log(6.4) / 27.0;
        const Scale = struct {
            htk: bool,
            log_step: f64,
            fn hzToMel(s: @This(), f_hz: f64) f64 {
                if (s.htk) return 2595.0 * libm.log10(1.0 + f_hz / 700.0);
                return if (f_hz < min_log_hz) f_hz * lin_slope else min_log_mel + libm.log(f_hz / min_log_hz) / s.log_step;
            }
            fn melToHz(s: @This(), m: f64) f64 {
                if (s.htk) return 700.0 * (pow10(m / 2595.0) - 1.0);
                return if (m < min_log_mel) m / lin_slope else min_log_hz * libm.exp((m - min_log_mel) * s.log_step);
            }
        };
        const sc: Scale = .{ .htk = use_htk, .log_step = log_step };

        // infer N_fft from n_fft_bins
        const bin_hz_step: f64 = @as(f64, @floatFromInt(sample_rate)) / @as(f64, @floatFromInt(n_fft));

        // mel grid: n_mel + 2 edges
        const m_lo = sc.hzToMel(fmin);
        const m_hi = sc.hzToMel(fmax);
        const n_pts: usize = @intCast(n_mel + 2);
        const mel_pts = try gpa.alloc(f64, n_pts);
        defer gpa.free(mel_pts);
        for (mel_pts, 0..) |*p, i| {
            p.* = @mulAdd(f64, m_hi - m_lo, @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n_mel + 1)), m_lo);
        }

        // convert to Hz
        const hz_pts = try gpa.alloc(f64, n_pts);
        defer gpa.free(hz_pts);
        for (hz_pts, mel_pts) |*h, m| h.* = sc.melToHz(m);

        const n_fft_bins: i64 = @divTrunc(n_fft, 2) + 1;

        // filterbank
        var out: std.ArrayList(f32) = .empty;
        try out.appendNTimes(gpa, 0, @intCast(n_mel * n_fft_bins));
        var m: usize = 0;
        while (m < n_mel) : (m += 1) {
            const f_left = hz_pts[m];
            const f_center = hz_pts[m + 1];
            const f_right = hz_pts[m + 2];

            const denom_l = maxF64(1e-30, f_center - f_left);
            const denom_r = maxF64(1e-30, f_right - f_center);
            const enorm: f64 = if (slaney_area_norm) (2.0 / maxF64(1e-30, f_right - f_left)) else 1.0;

            var k: c_int = 0;
            while (k < n_fft_bins) : (k += 1) {
                const f: f64 = @as(f64, @floatFromInt(k)) * bin_hz_step;
                var w: f64 = 0.0;
                if (f >= f_left and f <= f_center) {
                    w = (f - f_left) / denom_l;
                } else if (f > f_center and f <= f_right) {
                    w = (f_right - f) / denom_r;
                }
                out.items[m * @as(usize, @intCast(n_fft_bins)) + @as(usize, @intCast(k))] = @floatCast(w * enorm * @as(f64, scale));
            }
        }

        self.filters.n_mel = n_mel;
        self.filters.n_fft = n_fft;
        self.filters.data.deinit(gpa);
        self.filters.data = out;
    }
};

/// Ports `dft_impl` (mtmd-audio.cpp:139 @b1b770d90): the DFT, or the inverse
/// DFT scaled by `1/N`, over real or interleaved complex input.
fn dftImpl(comptime inverse: bool, comptime real_input: bool, cache: *const Cache, in: [*]const f32, N: c_int, out: [*]f32) void {
    const n_sin_cos_vals: c_int = @intCast(cache.sin_vals.items.len);
    const sin_cos_step = @divTrunc(n_sin_cos_vals, N);

    const sign: f32 = if (inverse) 1.0 else -1.0;
    const scale: f32 = if (inverse) (1.0 / @as(f32, @floatFromInt(N))) else 1.0;

    var k: c_int = 0;
    while (k < N) : (k += 1) {
        var re: f32 = 0;
        var im: f32 = 0;

        var n: c_int = 0;
        while (n < N) : (n += 1) {
            const idx: usize = @intCast(@rem(k *% n *% sin_cos_step, n_sin_cos_vals));
            const cos_val = cache.cos_vals.items[idx];
            const sin_val = cache.sin_vals.items[idx];
            const nu: usize = @intCast(n);

            if (real_input) {
                const in_re = in[nu];
                re = @mulAdd(f32, in_re, cos_val, re);
                im = @mulAdd(f32, sign * in_re, sin_val, im);
            } else {
                const in_re = in[nu * 2 + 0];
                const in_im = in[nu * 2 + 1];
                // (a + bi) * (cos + sign*i*sin) = (a*cos - sign*b*sin) + (sign*a*sin + b*cos)i
                re += @mulAdd(f32, in_re, cos_val, -(sign * in_im * sin_val));
                im += @mulAdd(f32, sign * in_re, sin_val, in_im * cos_val);
            }
        }

        const ku: usize = @intCast(k);
        out[ku * 2 + 0] = re * scale;
        out[ku * 2 + 1] = im * scale;
    }
}

/// Ports `fft_impl` (mtmd-audio.cpp:183 @b1b770d90): recursive radix-2
/// Cooley-Tukey, falling back to the DFT at an odd length. `in` is scratch
/// past the input and `out` past the `2N` results, as in the C++.
fn fftImpl(comptime inverse: bool, comptime real_input: bool, cache: *const Cache, in: [*]f32, N: c_int, out: [*]f32) void {
    ggmlAssert(N > 0, 184, "N > 0");
    const n_sin_cos_vals: c_int = @intCast(cache.sin_vals.items.len);

    if (N == 1) {
        out[0] = in[0];
        out[1] = if (real_input) 0.0 else in[1];
        return;
    }

    const half_N = @divTrunc(N, 2);
    if (N - half_N * 2 == 1) {
        // Odd N: fall back to DFT
        dftImpl(inverse, real_input, cache, in, N, out);
        return;
    }

    const Nu: usize = @intCast(N);
    const hu: usize = @intCast(half_N);

    // Split into even and odd
    if (real_input) {
        // Real input: stride is 1, copy only real values
        const even = in + Nu;
        for (0..hu) |i| even[i] = in[2 * i];
        const even_fft = out + 2 * Nu;
        fftImpl(inverse, true, cache, even, half_N, even_fft);

        const odd = even;
        for (0..hu) |i| odd[i] = in[2 * i + 1];
        const odd_fft = even_fft + Nu;
        fftImpl(inverse, true, cache, odd, half_N, odd_fft);
    } else {
        // Complex input: stride is 2, copy complex pairs
        const even = in + Nu * 2;
        for (0..hu) |i| {
            even[i * 2 + 0] = in[2 * i * 2 + 0];
            even[i * 2 + 1] = in[2 * i * 2 + 1];
        }
        const even_fft = out + 2 * Nu;
        fftImpl(inverse, false, cache, even, half_N, even_fft);

        const odd = even;
        for (0..hu) |i| {
            odd[i * 2 + 0] = in[(2 * i + 1) * 2 + 0];
            odd[i * 2 + 1] = in[(2 * i + 1) * 2 + 1];
        }
        const odd_fft = even_fft + Nu;
        fftImpl(inverse, false, cache, odd, half_N, odd_fft);
    }

    const even_fft = out + 2 * Nu;
    const odd_fft = even_fft + Nu;

    const sin_cos_step = @divTrunc(n_sin_cos_vals, N);

    const sign: f32 = if (inverse) 1.0 else -1.0;
    const scale: f32 = if (inverse) 0.5 else 1.0;

    for (0..hu) |k| {
        const idx: usize = k * @as(usize, @intCast(sin_cos_step)); // t = 2*M_PI*k/N
        const re = cache.cos_vals.items[idx];
        const im = sign * cache.sin_vals.items[idx];

        const re_odd = odd_fft[2 * k + 0];
        const im_odd = odd_fft[2 * k + 1];
        const e_re = even_fft[2 * k + 0];
        const e_im = even_fft[2 * k + 1];

        out[2 * k + 0] = scale * @mulAdd(f32, -im, im_odd, @mulAdd(f32, re, re_odd, e_re));
        out[2 * k + 1] = scale * @mulAdd(f32, im, re_odd, @mulAdd(f32, re, im_odd, e_im));

        out[2 * (k + hu) + 0] = scale * @mulAdd(f32, im, im_odd, @mulAdd(f32, -re, re_odd, e_re));
        out[2 * (k + hu) + 1] = scale * @mulAdd(f32, -im, re_odd, @mulAdd(f32, -re, im_odd, e_im));
    }
}

/// Ports `fft` (mtmd-audio.cpp:264 @b1b770d90): forward, real input.
fn fft(cache: *const Cache, in: [*]f32, N: c_int, out: [*]f32) void {
    fftImpl(false, true, cache, in, N, out);
}

/// Ports `ifft` (mtmd-audio.cpp:269 @b1b770d90): inverse, complex input.
fn ifft(cache: *const Cache, in: [*]f32, N: c_int, out: [*]f32) void {
    fftImpl(true, false, cache, in, N, out);
}

/// Ports `mtmd_audio_mel_config` (mtmd-audio.h:57 @b1b770d90): what the
/// log-mel graph and its padding read, filled by `initialize` of the
/// preprocessors that use it.
///
/// The C++'s `cache` pointer is `has_cache` here: it points into the
/// preprocessor that owns the config, and a Zig `Preprocessor` is a value
/// its owner may move, so the cache is passed beside the config instead.
pub const MelConfig = struct {
    n_mel: i64 = 0,
    n_fft_bins: i64 = 0,
    hann_window_size: i32 = 0,
    hop_length: i32 = 0,
    sample_rate: i32 = 0,
    no_padding: bool = false,
    center_padding: bool = false,
    preemph: f32 = 0.0,
    use_natural_log: bool = false,
    norm_per_feature: bool = false,
    use_magnitude: bool = false, // |X| instead of |X|^2
    mel_floor: f32 = 5.960464477539063e-08,
    mel_floor_add: bool = false, // log(x + floor) instead of log(max(x, floor))
    std_eps_after_sqrt: bool = false, // std + eps instead of sqrt(var + eps)
    n_threads: c_int = 1,

    /// The C++'s `cache != nullptr`: the preprocessor's own `cache` is the
    /// log-mel's, and `preprocess` defers the graph to `compute`.
    has_cache: bool = false,
    /// must be set before the first `compute`, see `mtmd_context::init_audio()`
    sched: c.ggml_backend_sched_t = null,
};

/// Ports `mtmd_backend_cpu_set_n_threads` (mtmd-subcontext.cpp:24 @b1b770d90).
pub fn backendCpuSetNThreads(backend_cpu: c.ggml_backend_t, n_threads: c_int) void {
    const dev = c.ggml_backend_get_device(backend_cpu);
    const reg: c.ggml_backend_reg_t = if (dev != null) c.ggml_backend_dev_backend_reg(dev) else null;
    if (reg != null) {
        const p = c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (p != null) {
            const set_n_threads: c.ggml_backend_set_n_threads_t = @ptrCast(@alignCast(p));
            set_n_threads.?(backend_cpu, n_threads);
        }
    }
}

/// `t->nb[1]` of a tensor known not to be null.
fn tensorNb1(t: ?*c.ggml_tensor) usize {
    return t.?.nb[1];
}

/// Ports `mtmd_audio_log_mel` (mtmd-subcontext.cpp:274 @b1b770d90): the
/// log-mel of `n_frames` frames as a ggml graph on `config.sched`. The
/// waveform must already be padded and `window` must hold one whole frame;
/// `out` has the layout `[n_frames, n_mel]`. Must not run at the same time
/// as another pass on the scheduler.
///
/// Return: false, logged, when the graph cannot be allocated or computed.
pub fn logMel(config: *const MelConfig, cache: *const Cache, samples: [*]const f32, window: [*]const f32, n_frames: i64, out: [*]f32) !bool {
    const sched = config.sched;
    const backend_cpu = c.ggml_backend_sched_get_backend(sched, c.ggml_backend_sched_get_n_backends(sched) - 1);
    backendCpuSetNThreads(backend_cpu, @max(1, config.n_threads));

    const filters = cache.filters.data.items;
    const n_fft_bins = config.n_fft_bins;
    const frame_size = (n_fft_bins - 1) * 2;
    const n_mel = config.n_mel;
    impl.ggmlAssert(filters.len == @as(usize, @intCast(n_mel * n_fft_bins)), "mtmd-subcontext.cpp", 284, "filters.size() == (size_t) (n_mel * n_fft_bins)");
    impl.ggmlAssert(cache.cos_vals.items.len == @as(usize, @intCast(frame_size)) and cache.sin_vals.items.len == @as(usize, @intCast(frame_size)), "mtmd-subcontext.cpp", 285, "cache.cos_vals.size() == (size_t) frame_size && cache.sin_vals.size() == (size_t) frame_size");

    // DFT matrix with the window folded in, the first n_fft_bins rows give the real part
    const fs: usize = @intCast(frame_size);
    const nb: usize = @intCast(n_fft_bins);
    const dft = try gpa.alloc(f32, 2 * nb * fs);
    defer gpa.free(dft);
    for (0..nb) |k| {
        for (0..fs) |j| {
            const q = (j * k) % fs;
            dft[k * fs + j] = window[j] * cache.cos_vals.items[q];
            dft[(k + nb) * fs + j] = window[j] * cache.sin_vals.items[q];
        }
    }

    // the spectrum is several times larger than the result, so long audio is processed in blocks
    const n_frames_max: i64 = 8192;

    var off: i64 = 0;
    while (off < n_frames) : (off += n_frames_max) {
        const n_cur = @min(n_frames_max, n_frames - off);

        const params: c.ggml_init_params = .{
            .mem_size = c.ggml_tensor_overhead() * 32 + c.ggml_graph_overhead(),
            .mem_buffer = null,
            .no_alloc = true,
        };
        const ctx0 = c.ggml_init(params);
        defer c.ggml_free(ctx0);

        const inp_samples = c.ggml_new_tensor_1d(ctx0, c.GGML_TYPE_F32, (n_cur - 1) * config.hop_length + frame_size);
        const inp_dft = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, frame_size, 2 * n_fft_bins);
        const inp_filters = c.ggml_new_tensor_2d(ctx0, c.GGML_TYPE_F32, n_fft_bins, n_mel);
        c.ggml_set_input(inp_samples);
        c.ggml_set_input(inp_dft);
        c.ggml_set_input(inp_filters);

        var cur = c.ggml_cont(ctx0, c.ggml_view_2d(ctx0, inp_samples, frame_size, n_cur, @as(usize, @intCast(config.hop_length)) * c.ggml_element_size(inp_samples), 0));
        cur = c.ggml_mul_mat(ctx0, inp_dft, cur); // [2 * n_fft_bins, n_cur]
        _ = c.ggml_prec_set_src(cur, c.GGML_PREC_F32, 1);

        cur = c.ggml_sqr(ctx0, cur);
        const nb1 = tensorNb1(cur);
        cur = c.ggml_add(ctx0, c.ggml_view_2d(ctx0, cur, n_fft_bins, n_cur, nb1, 0), c.ggml_view_2d(ctx0, cur, n_fft_bins, n_cur, nb1, nb * c.ggml_element_size(cur)));
        if (config.use_magnitude) {
            cur = c.ggml_sqrt(ctx0, cur);
        }

        cur = c.ggml_mul_mat(ctx0, cur, inp_filters); // [n_cur, n_mel]
        _ = c.ggml_prec_set_src(cur, c.GGML_PREC_F32, 1);
        cur = if (config.mel_floor_add)
            c.ggml_scale_bias(ctx0, cur, 1.0, config.mel_floor)
        else
            c.ggml_clamp(ctx0, cur, config.mel_floor, std.math.floatMax(f32));
        cur = c.ggml_log(ctx0, cur);
        if (!config.use_natural_log) {
            // clang folds `1.0f / logf(10.0f)` at compile time, and LLVM
            // folds `logf` of a constant through the host's `double` `log`
            cur = c.ggml_scale(ctx0, cur, 1.0 / @as(f32, @floatCast(libm.log(10.0))));
        }
        c.ggml_set_output(cur);

        const gf = c.ggml_new_graph(ctx0);
        c.ggml_build_forward_expand(gf, cur);

        c.ggml_backend_sched_reset(sched);
        if (!c.ggml_backend_sched_alloc_graph(sched, gf)) {
            impl.logErr("%s: failed to allocate compute graph\n", .{"mtmd_audio_log_mel"});
            return false;
        }

        c.ggml_backend_tensor_set(inp_samples, samples + @as(usize, @intCast(off * config.hop_length)), 0, c.ggml_nbytes(inp_samples));
        c.ggml_backend_tensor_set(inp_dft, dft.ptr, 0, c.ggml_nbytes(inp_dft));
        c.ggml_backend_tensor_set(inp_filters, filters.ptr, 0, c.ggml_nbytes(inp_filters));

        const status = c.ggml_backend_sched_graph_compute(sched, gf);
        if (status != c.GGML_STATUS_SUCCESS) {
            impl.logErr("%s: ggml_backend_sched_graph_compute failed with error %d\n", .{ "mtmd_audio_log_mel", @as(c_int, status) });
            return false;
        }

        const cur_nb1 = tensorNb1(cur);
        var j: i64 = 0;
        while (j < n_mel) : (j += 1) {
            c.ggml_backend_tensor_get(cur, out + @as(usize, @intCast(j * n_frames + off)), @as(usize, @intCast(j)) * cur_nb1, @as(usize, @intCast(n_cur)) * c.ggml_element_size(cur));
        }
    }

    return true;
}

/// `sum += (x[j] - mean) * (x[j] - mean)` over a `float` row into a
/// `double`, **as the reference compiler emits it**: clang marks the
/// expression `fmuladd`, and the loop vectorizer turns the loop into a strict
/// in-order reduction -- products rounded (`fmul.2d`) in whole groups of four,
/// added one at a time -- leaving only the `t % 4` past the last group scalar
/// and fused (`fmadd`). Read off the reference `mtmd-audio.o`'s disassembly,
/// in both `log_mel_spectrogram` and the parakeet `preprocess`; the shape of
/// "one expression is fused; one *reduction loop* is not". The
/// gate cannot tell the two apart here -- the variance only reaches a float
/// through a square root and a division -- so the disassembly is the evidence.
pub fn strictSquareSum(row: []const f32, mean: f64) f64 {
    var sum: f64 = 0.0;
    const split = row.len - row.len % 4;
    for (row[0..split]) |v| {
        const d: f64 = @as(f64, v) - mean;
        sum += d * d;
    }
    for (row[split..]) |v| {
        const d: f64 = @as(f64, v) - mean;
        sum = @mulAdd(f64, d, d, sum);
    }
    return sum;
}

/// Ports `mtmd_audio_log_mel_frame` (mtmd-audio.cpp:313 @b1b770d90): one
/// log10-mel frame from `n_fft` windowed samples; `cache` must be initialized
/// for `n_fft`.
pub fn logMelFrame(cache: *const Cache, samples: [*]const f32, n_fft: c_int, n_mel: c_int, out: [*]f32) !void {
    const n_fft_bins: usize = @intCast(@divTrunc(n_fft, 2) + 1);
    ggmlAssert(cache.hann_window.items.len == @as(usize, @intCast(n_fft)), 315, "(int) cache.hann_window.size() == n_fft");
    ggmlAssert(cache.filters.n_mel == n_mel and cache.filters.n_fft == n_fft, 316, "cache.filters.n_mel == n_mel && cache.filters.n_fft == n_fft");

    const nf: usize = @intCast(n_fft);
    const fft_in = try gpa.alloc(f32, nf * 2);
    defer gpa.free(fft_in);
    @memset(fft_in, 0.0);
    const fft_out = try gpa.alloc(f32, nf * 2 * 2 * 2);
    defer gpa.free(fft_out);
    @memset(fft_out, 0.0);
    for (0..nf) |j| fft_in[j] = cache.hann_window.items[j] * samples[j];
    fft(cache, fft_in.ptr, n_fft, fft_out.ptr);

    for (0..n_fft_bins) |j| {
        const a = fft_out[2 * j + 0];
        const b = fft_out[2 * j + 1];
        fft_out[j] = @mulAdd(f32, a, a, b * b);
    }

    for (0..@as(usize, @intCast(n_mel))) |j| {
        var sum: f64 = 0.0;
        for (0..n_fft_bins) |k| {
            sum += fft_out[k] * cache.filters.data.items[j * n_fft_bins + k];
        }
        out[j] = @floatCast(libm.log10(maxF64(sum, 1e-10)));
    }
}

/// Ports `log_mel_spectrogram` (mtmd-audio.cpp:339 @b1b770d90), after
/// openai/whisper's `audio.py`. `shape_only` skips the log-mel graph: the
/// shapes are right and the data is zero-filled.
fn logMelSpectrogram(samples_in: []const f32, params: *const MelConfig, cache: *const Cache, shape_only: bool, out: *Mel) !bool {
    out.n_len_org = @intCast(samples_in.len);
    var n_samples: c_int = @intCast(samples_in.len);

    // Hann window
    var hann: [*]const f32 = cache.hann_window.items.ptr;
    const frame_size: c_int = @intCast((params.n_fft_bins - 1) * 2);
    const frame_step = params.hop_length;

    // Padding
    var samples_padded: std.ArrayList(f32) = .empty;
    defer samples_padded.deinit(gpa);
    if (params.no_padding) {
        // no padding, use samples as-is
        try samples_padded.appendSlice(gpa, samples_in);
        n_samples = @intCast(samples_padded.items.len);
    } else if (params.center_padding) {
        const pad_amount: usize = @intCast(@divTrunc(frame_size, 2));
        try samples_padded.appendNTimes(gpa, 0, samples_in.len + 2 * pad_amount);
        @memcpy(samples_padded.items[pad_amount..][0..samples_in.len], samples_in);
        n_samples = @intCast(samples_padded.items.len);
    } else {
        // existing padding logic
        const stage_1_pad: usize = @intCast(@as(i64, params.sample_rate) * 30);
        const stage_2_pad: usize = @intCast(@divTrunc(frame_size, 2));
        const ns: usize = @intCast(n_samples);
        try samples_padded.appendNTimes(gpa, 0, ns + stage_1_pad + stage_2_pad * 2);
        @memcpy(samples_padded.items[stage_2_pad..][0..ns], samples_in[0..ns]);
        // 30 seconds of zeros at the end (already zero), then the reflection
        // at the beginning
        if (ns < stage_2_pad + 1) return false;
        for (0..stage_2_pad) |k| samples_padded.items[k] = samples_in[stage_2_pad - k];

        n_samples = @intCast(samples_padded.items.len);
    }

    // preemphasis
    if (params.preemph != 0) {
        const pad_amount = @divTrunc(frame_size, 2);
        const preemph: f32 = 0.97;
        const sp = samples_padded.items;
        var prev = sp[@intCast(pad_amount)];
        var i: c_int = pad_amount + 1;
        while (i + pad_amount < n_samples) : (i += 1) {
            const cur = sp[@intCast(i)];
            sp[@intCast(i)] = @mulAdd(f32, -preemph, prev, cur);
            prev = cur;
        }
    }

    // pad hann window if it's smaller than frame_size
    var hann_window_padded: std.ArrayList(f32) = .empty;
    defer hann_window_padded.deinit(gpa);
    if (params.hann_window_size < frame_size) {
        try hann_window_padded.appendNTimes(gpa, 0, @intCast(frame_size));
        const padding: usize = @intCast(@divTrunc(frame_size - params.hann_window_size, 2));
        const hw: usize = @intCast(params.hann_window_size);
        @memcpy(hann_window_padded.items[padding..][0..hw], hann[0..hw]);
        hann = hann_window_padded.items.ptr;
    }

    ggmlAssert(params.n_fft_bins > 0, 414, "params.n_fft_bins > 0");
    ggmlAssert(params.hop_length > 0, 415, "params.hop_length > 0");
    out.n_mel = params.n_mel;
    out.n_len = @divTrunc(n_samples - frame_size, frame_step) + 1;
    // Validate dimensions before allocation to prevent integer overflow
    if (out.n_mel <= 0 or out.n_len <= 0) {
        impl.logErr("%s: invalid mel dimensions n_mel=%lld n_len=%lld\n", .{ "log_mel_spectrogram", @as(c_longlong, out.n_mel), @as(c_longlong, out.n_len) });
        return false;
    }
    const total_size: usize = @as(usize, @intCast(out.n_mel)) *% @as(usize, @intCast(out.n_len));
    if (total_size > std.math.maxInt(usize) / @sizeOf(f32)) {
        impl.logErr("%s: size overflow: n_mel=%lld n_len=%lld\n", .{ "log_mel_spectrogram", @as(c_longlong, out.n_mel), @as(c_longlong, out.n_len) });
        return false;
    }
    if (n_samples < frame_size) {
        impl.logErr("%s: not enough samples after padding\n", .{"log_mel_spectrogram"});
        return false;
    }
    const old_len = out.data.items.len;
    try out.data.resize(gpa, total_size);
    if (total_size > old_len) @memset(out.data.items[old_len..], 0.0);

    if (!shape_only and !try logMel(params, cache, samples_padded.items.ptr, hann, out.n_len, out.data.items.ptr)) {
        return false;
    }

    const effective_n_len: i64 = @divTrunc(@as(i64, @intCast(samples_in.len)), frame_step);
    const n_len_u: usize = @intCast(out.n_len);
    const d = out.data.items;
    if (params.norm_per_feature) {
        ggmlAssert(effective_n_len > 1, 440, "effective_n_len > 1");
        const eff: usize = @intCast(effective_n_len);
        var i: usize = 0;
        while (i < out.n_mel) : (i += 1) {
            const row = d[i * n_len_u ..];
            var mean: f64 = 0;
            for (row[0..eff]) |v| mean += v;
            mean /= @floatFromInt(effective_n_len);

            var var_ = strictSquareSum(row[0..eff], mean);
            var_ /= @floatFromInt(effective_n_len - 1); // unbiased
            const mstd = if (params.std_eps_after_sqrt) @sqrt(var_) + 1e-5 else @sqrt(var_ + 1e-5);

            for (row[0..eff]) |*v| v.* = @floatCast((@as(f64, v.*) - mean) / mstd);

            // pad the rest with zeros
            @memset(row[eff..n_len_u], 0.0);
        }
    } else if (!params.no_padding) {
        // Whisper-style clamping and normalization (NOT used by Gemma4)
        var mmax: f64 = -1e20;
        for (d) |v| {
            if (v > mmax) mmax = v;
        }

        mmax -= 8.0;

        for (d) |*v| {
            if (v.* < mmax) v.* = @floatCast(mmax);
            v.* = @floatCast((@as(f64, v.*) + 4.0) / 4.0);
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// chatterbox's reference front-end

/// `sin(x)` and `cos(x)` in `double` as clang emits the pair: Apple's
/// `__sincos_stret` on Darwin (measured from the reference `mtmd-audio.o`'s
/// imports), the separate calls under `-fmath-errno`.
fn sinCosF64(x: f64) [2]f64 {
    if (comptime !math_errno) {
        const r = libm.__sincos_stret(x);
        return .{ r.sin, r.cos };
    }
    return .{ libm.sin(x), libm.cos(x) };
}

/// `powf(10.0f, x)` as clang emits it: `__exp10f` on Darwin (measured from
/// the reference `mtmd-audio.o`'s imports).
fn pow10f(x: f32) f32 {
    if (comptime !math_errno) return libm.__exp10f(x);
    return libm.powf(10.0, x);
}

/// `std::max(a, b)` over `float`: `(a < b) ? b : a`.
fn maxF32(a: f32, b: f32) f32 {
    return if (a < b) b else a;
}

/// The FFT buffers upstream's chatterbox helpers allocate per call:
/// `fft_in` of `2 * n_fft` and `fft_out` of `8 * n_fft`, zeroed.
const CbxFftBufs = struct {
    in: []f32,
    out: []f32,

    fn init(n_fft: usize) !CbxFftBufs {
        const in = try gpa.alloc(f32, n_fft * 2);
        errdefer gpa.free(in);
        const out = try gpa.alloc(f32, n_fft * 8);
        @memset(in, 0.0);
        @memset(out, 0.0);
        return .{ .in = in, .out = out };
    }

    fn deinit(self: *CbxFftBufs) void {
        gpa.free(self.in);
        gpa.free(self.out);
    }
};

/// Ports `mtmd_audio_reflect_frame` (mtmd-audio.cpp:910 @b1b770d90): a
/// windowed frame of a centered, reflect padded signal, `dst[i] = x[start +
/// i] * win[i]`, zeros past the window.
fn cbxReflectFrame(x: [*]const f32, n: c_int, start: c_int, win: []const f32, dst: []f32) void {
    @memset(dst, 0.0);
    for (win, 0..) |w, iu| {
        var idx: c_int = start + @as(c_int, @intCast(iu));
        if (idx < 0) {
            idx = -idx;
        }
        if (idx >= n) {
            idx = 2 * n - 2 - idx;
        }
        dst[iu] = x[@intCast(idx)] * w;
    }
}

/// Ports `mtmd_audio_mel_energies` (mtmd-audio.cpp:925 @b1b770d90): mel
/// energies of the frame in `fft_in`, the power (or magnitude) spectrum
/// through `cache.filters`, into `mel`.
fn cbxMelEnergies(cache: *const Cache, fft_in: []f32, fft_out: []f32, magnitude: bool, mel: *std.ArrayList(f64)) !void {
    const n_fft: c_int = @intCast(cache.filters.n_fft);
    const n_bins: usize = @intCast(@divTrunc(n_fft, 2) + 1);
    fft(cache, fft_in.ptr, n_fft, fft_out.ptr);
    for (0..n_bins) |k| {
        const re = fft_out[2 * k + 0];
        const im = fft_out[2 * k + 1];
        const p = @mulAdd(f32, re, re, im * im);
        fft_out[k] = if (magnitude) @sqrt(p + 1e-9) else p;
    }
    const n_mel: usize = @intCast(cache.filters.n_mel);
    mel.clearRetainingCapacity();
    try mel.appendNTimes(gpa, 0.0, n_mel);
    for (0..n_mel) |m| {
        const w = cache.filters.data.items[m * n_bins ..][0..n_bins];
        var acc = mel.items[m];
        // a `float` product into a `double` sum: never contracted
        for (w, fft_out[0..n_bins]) |wk, fk| acc += @as(f64, wk * fk);
        mel.items[m] = acc;
    }
}

/// Ports `mtmd_audio_s3tok_log_mel` (mtmd-audio.cpp:944 @b1b770d90): the
/// whisper style log-mel of the chatterbox s3 tokenizer (s3tokenizer.py),
/// mel major.
fn cbxS3tokLogMel(cache: *const Cache, samples: []const f32, out: *std.ArrayList(f32), n_frames: *c_int) !bool {
    const n_fft: c_int = 400;
    const hop: c_int = 160;
    const n_mel: usize = @intCast(cache.filters.n_mel);
    const n: c_int = @intCast(samples.len);

    n_frames.* = @divTrunc(n, hop);
    if (n_frames.* <= 0) {
        return false;
    }
    const nf: usize = @intCast(n_frames.*);

    var bufs = try CbxFftBufs.init(n_fft);
    defer bufs.deinit();
    var e: std.ArrayList(f64) = .empty;
    defer e.deinit(gpa);
    const mel = try gpa.alloc(f64, n_mel * nf);
    defer gpa.free(mel);
    @memset(mel, 0.0);
    for (0..nf) |fru| {
        const fr: c_int = @intCast(fru);
        cbxReflectFrame(samples.ptr, n, fr * hop - @divTrunc(n_fft, 2), cache.hann_window.items, bufs.in);
        try cbxMelEnergies(cache, bufs.in, bufs.out, false, &e);
        for (0..n_mel) |m| {
            mel[m * nf + fru] = libm.log10(maxF64(e.items[m], 1e-10));
        }
    }

    // std::max_element: the first of the largest
    var mx = mel[0];
    for (mel[1..]) |v| {
        if (mx < v) mx = v;
    }
    out.clearRetainingCapacity();
    try out.resize(gpa, mel.len);
    for (out.items, mel) |*o, v| {
        o.* = @floatCast((maxF64(v, mx - 8.0) + 4.0) / 4.0);
    }
    return true;
}

/// Ports `mtmd_audio_upsample_3_2` (mtmd-audio.cpp:976 @b1b770d90): 16 kHz
/// to 24 kHz, a hann windowed sinc cut just under the source nyquist.
fn cbxUpsample32(samples: []const f32, out: *std.ArrayList(f32)) !void {
    const W: c_int = 16; // sinc half width in source samples
    const fc: f64 = 0.495; // cutoff, normalized to the source rate

    const n_out = samples.len * 3 / 2;
    const n_in: c_int = @intCast(samples.len);
    out.clearRetainingCapacity();
    try out.appendNTimes(gpa, 0.0, n_out);
    for (0..n_out) |n| {
        const t: f64 = @as(f64, @floatFromInt(2 * n)) / 3.0;
        const k0: c_int = @as(c_int, @intFromFloat(@floor(t))) - W + 1;
        var acc: f64 = 0.0;
        var k = k0;
        while (k < k0 + 2 * W) : (k += 1) {
            if (k < 0 or k >= n_in) {
                continue;
            }
            const x = t - @as(f64, @floatFromInt(k));
            const s: f64 = if (x == 0.0) 1.0 else libm.sin(2.0 * m_pi * fc * x) / (m_pi * x);
            acc = @mulAdd(f64, @as(f64, samples[@intCast(k)]) * s * 0.5, 1.0 + libm.cos(m_pi * x / @as(f64, W + 1)), acc);
        }
        out.items[n] = @floatCast(acc);
    }
}

/// Ports `mtmd_audio_matcha_log_mel` (mtmd-audio.cpp:999 @b1b770d90): the
/// matcha style log-mel of the s3gen prompt features (s3gen/utils/mel.py),
/// frame major.
fn cbxMatchaLogMel(cache: *const Cache, samples: []const f32, out: *std.ArrayList(f32), n_frames: *c_int) !bool {
    const n_fft: c_int = 1920;
    const hop: c_int = 480;
    const pad: c_int = @divTrunc(n_fft - hop, 2);
    const n_mel: usize = 80;
    const n: c_int = @intCast(samples.len);

    n_frames.* = 1 + @divTrunc(n + 2 * pad - n_fft, hop);
    if (n <= pad or n_frames.* <= 0) {
        return false;
    }
    const nf: usize = @intCast(n_frames.*);

    var bufs = try CbxFftBufs.init(n_fft);
    defer bufs.deinit();
    var e: std.ArrayList(f64) = .empty;
    defer e.deinit(gpa);
    out.clearRetainingCapacity();
    try out.resize(gpa, n_mel * nf);
    for (0..nf) |fru| {
        const fr: c_int = @intCast(fru);
        cbxReflectFrame(samples.ptr, n, fr * hop - pad, cache.hann_window.items, bufs.in);
        try cbxMelEnergies(cache, bufs.in, bufs.out, true, &e);
        for (0..n_mel) |m| {
            out.items[fru * n_mel + m] = libm.logf(maxF32(@floatCast(e.items[m]), 1e-5));
        }
    }
    return true;
}

/// Ports `mtmd_audio_trim_silence` (mtmd-audio.cpp:1026 @b1b770d90):
/// librosa.effects.trim, non-silent where the rms window is within `top_db`
/// of the peak.
fn cbxTrimSilence(samples: []const f32, top_db: f32, start: *usize, end: *usize) !void {
    const win: c_int = 2048;
    const hop: c_int = 512;
    const n: c_int = @intCast(samples.len);

    const n_fr: c_int = 1 + @divTrunc(n, hop);
    const rms = try gpa.alloc(f64, @intCast(n_fr));
    defer gpa.free(rms);
    var mx: f64 = 0.0;
    for (rms, 0..) |*r, fru| {
        const fr: c_int = @intCast(fru);
        var acc: f64 = 0.0;
        var i: c_int = 0;
        while (i < win) : (i += 1) {
            const idx = fr * hop - @divTrunc(win, 2) + i;
            if (idx >= 0 and idx < n) {
                const v: f64 = samples[@intCast(idx)];
                acc = @mulAdd(f64, v, v, acc);
            }
        }
        r.* = @sqrt(acc / @as(f64, win));
        mx = maxF64(mx, r.*);
    }

    // the call is inlined with a constant `top_db`, so clang folds the pow
    // through the host's `pow`, not `__exp10`
    const thr = mx * libm.pow(10.0, -@as(f64, top_db) / 20.0);
    var first: c_int = -1;
    var last: c_int = -1;
    for (rms, 0..) |r, fru| {
        if (r > thr) {
            if (first < 0) {
                first = @intCast(fru);
            }
            last = @intCast(fru);
        }
    }
    if (first < 0) {
        start.* = 0;
        end.* = 0;
        return;
    }
    start.* = @as(usize, @intCast(first)) * @as(usize, @intCast(hop));
    end.* = @min(@as(usize, @intCast(last + 1)) * @as(usize, @intCast(hop)), samples.len);
}

/// Ports `mtmd_audio_ve_mel` (mtmd-audio.cpp:1066 @b1b770d90): the power mel
/// of the voice encoder front-end, frame major, no log.
fn cbxVeMel(cache: *const Cache, samples: []const f32, out: *std.ArrayList(f32), n_frames: *c_int) !bool {
    const n_fft: c_int = 400;
    const hop: c_int = 160;
    const n_mel: usize = 40;
    const n: c_int = @intCast(samples.len);

    n_frames.* = 1 + @divTrunc(n, hop);
    if (n < 2) {
        return false;
    }
    const nf: usize = @intCast(n_frames.*);

    var bufs = try CbxFftBufs.init(n_fft);
    defer bufs.deinit();
    var e: std.ArrayList(f64) = .empty;
    defer e.deinit(gpa);
    out.clearRetainingCapacity();
    try out.resize(gpa, n_mel * nf);
    for (0..nf) |fru| {
        const fr: c_int = @intCast(fru);
        cbxReflectFrame(samples.ptr, n, fr * hop - @divTrunc(n_fft, 2), cache.hann_window.items, bufs.in);
        try cbxMelEnergies(cache, bufs.in, bufs.out, false, &e);
        for (0..n_mel) |m| {
            out.items[fru * n_mel + m] = @floatCast(e.items[m]);
        }
    }
    return true;
}

/// A lambda of `mtmd_audio_lufs`, its `biquad` (mtmd-audio.cpp:1095
/// @b1b770d90), in place over `y`.
fn cbxBiquad(y: []f64, b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) void {
    var x1: f64 = 0.0;
    var x2: f64 = 0.0;
    var y1: f64 = 0.0;
    var y2: f64 = 0.0;
    for (y) |*v| {
        const x0 = v.*;
        // b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2, as clang contracts it
        v.* = @mulAdd(f64, -a2, y2, @mulAdd(f64, -a1, y1, @mulAdd(f64, b2, x2, @mulAdd(f64, b0, x0, b1 * x1))));
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = v.*;
    }
}

/// A lambda of `mtmd_audio_lufs`, its `gated_mean` (mtmd-audio.cpp:1149
/// @b1b770d90).
fn cbxGatedMean(z: []const f64, thr_lufs: f64) f64 {
    var acc: f64 = 0.0;
    var n: c_int = 0;
    for (z) |v| {
        if (@mulAdd(f64, 10.0, libm.log10(maxF64(v, 1e-30)), -0.691) > thr_lufs) {
            acc += v;
            n += 1;
        }
    }
    return if (n > 0) acc / @as(f64, @floatFromInt(n)) else 0.0;
}

/// Ports `mtmd_audio_lufs` (mtmd-audio.cpp:1092 @b1b770d90): the ITU-R
/// BS.1770 integrated loudness of a mono signal, matching pyloudnorm.
fn cbxLufs(samples: []const f32, sample_rate: c_int) !f32 {
    const y = try gpa.alloc(f64, samples.len);
    defer gpa.free(y);
    for (y, samples) |*d, v| d.* = v;
    const sr: f64 = @floatFromInt(sample_rate);

    // stage 1: high shelf, f0 1681.9744509555319 Hz, +3.99984385397 dB, Q 0.7071752369554196
    {
        // a constant argument: clang folds the pow through the host's `pow`
        const A = libm.pow(10.0, @as(f64, 3.99984385397) / 40.0);
        const w0 = 2.0 * m_pi * 1681.9744509555319 / sr;
        const sc = sinCosF64(w0);
        const alpha = sc[0] / (2.0 * 0.7071752369554196);
        const cw = sc[1];
        const sq = 2.0 * @sqrt(A) * alpha;
        const b0 = A * (@mulAdd(f64, A - 1.0, cw, A + 1.0) + sq);
        const b1 = -2.0 * A * @mulAdd(f64, A + 1.0, cw, A - 1.0);
        const b2 = A * (@mulAdd(f64, A - 1.0, cw, A + 1.0) - sq);
        const a0 = @mulAdd(f64, -(A - 1.0), cw, A + 1.0) + sq;
        const a1 = 2.0 * @mulAdd(f64, -(A + 1.0), cw, A - 1.0);
        const a2 = @mulAdd(f64, -(A - 1.0), cw, A + 1.0) - sq;
        cbxBiquad(y, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
    }
    // stage 2: high pass, f0 38.13547087602444 Hz, Q 0.5003270373238773
    {
        const w0 = 2.0 * m_pi * 38.13547087602444 / sr;
        const sc = sinCosF64(w0);
        const alpha = sc[0] / (2.0 * 0.5003270373238773);
        const cw = sc[1];
        const b0 = (1.0 + cw) / 2.0;
        const b1 = -(1.0 + cw);
        const b2 = (1.0 + cw) / 2.0;
        const a0 = 1.0 + alpha;
        const a1 = -2.0 * cw;
        const a2 = 1.0 - alpha;
        cbxBiquad(y, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
    }

    const win: c_int = @intFromFloat(0.4 * sr);
    const hop: c_int = @divTrunc(win, 4);
    const n: c_int = @intCast(samples.len);
    if (n < win) {
        return -std.math.inf(f32);
    }
    const n_blocks: c_int = 1 + @divTrunc(n - win, hop);
    const z = try gpa.alloc(f64, @intCast(n_blocks));
    defer gpa.free(z);
    const wu: usize = @intCast(win);
    for (z, 0..) |*zb, b| {
        const blk = y[b * @as(usize, @intCast(hop)) ..][0..wu];
        // the strict in-order reduction clang's loop vectorizer makes of
        // `acc += y * y`: products rounded (`fmul.2d`) in whole groups of
        // eight, only the `win % 8` past the last group fused (`fmadd`), read
        // off the reference `mtmd-audio.o`'s disassembly at -O2, baseline
        // CPU. The gate cannot tell: `win` is 6400 at 16 kHz
        var acc: f64 = 0.0;
        const split = wu - wu % 8;
        for (blk[0..split]) |v| acc += v * v;
        for (blk[split..]) |v| acc = @mulAdd(f64, v, v, acc);
        zb.* = acc / @as(f64, @floatFromInt(win));
    }

    const z_abs = cbxGatedMean(z, -70.0);
    if (z_abs <= 0.0) {
        return -std.math.inf(f32);
    }
    const thr_rel = @mulAdd(f64, 10.0, libm.log10(z_abs), -0.691) - 10.0;
    const z_rel = cbxGatedMean(z, thr_rel);
    if (z_rel <= 0.0) {
        return -std.math.inf(f32);
    }
    return @floatCast(@mulAdd(f64, 10.0, libm.log10(z_rel), -0.691));
}

/// A lambda of `mtmd_audio_preprocessor_chatterbox_ref`'s `initialize`,
/// its `mel` (mtmd-audio.cpp:1192 @b1b770d90): kaldi's mel scale.
fn cbxKaldiMel(f: f64) f64 {
    return 1127.0 * libm.log(1.0 + f / 700.0);
}

/// Ports `initialize` (mtmd-audio.cpp:1177 @b1b770d90), a member of
/// `mtmd_audio_preprocessor_chatterbox_ref`. `self.cache` is its
/// `fbank_cache`.
fn cbxRefInitialize(self: *Preprocessor) !void {
    const hp = self.hparams;
    // kaldi fbank: povey window, kaldi mel banks over the 512-point spectrum, the nyquist bin left out
    {
        const frame_len: c_int = 400;
        const n_fft: c_int = 512;
        const n_bins: usize = @intCast(@divTrunc(n_fft, 2) + 1);
        const n_mel: c_int = hp.n_mel_bins;
        const sr: f64 = @floatFromInt(hp.audio_sample_rate);
        const fc = &self.cache;

        try fc.fillSinCosTable(n_fft);
        try fc.hann_window.resize(gpa, frame_len);
        for (fc.hann_window.items, 0..) |*w, i| {
            const arg = 2.0 * m_pi * @as(f64, @floatFromInt(i)) / @as(f64, frame_len - 1);
            w.* = @floatCast(libm.pow(@mulAdd(f64, -0.5, libm.cos(arg), 0.5), 0.85));
        }

        const mel_lo = cbxKaldiMel(20.0);
        const mel_hi = cbxKaldiMel(sr / 2.0);
        const delta = (mel_hi - mel_lo) / @as(f64, @floatFromInt(n_mel + 1));
        const bin_hz = sr / @as(f64, n_fft);

        const filters = &fc.filters;
        filters.n_mel = n_mel;
        filters.n_fft = n_fft;
        filters.data.clearRetainingCapacity();
        try filters.data.appendNTimes(gpa, 0.0, @as(usize, @intCast(n_mel)) * n_bins);
        for (0..@intCast(n_mel)) |m| {
            const left = @mulAdd(f64, @floatFromInt(m), delta, mel_lo);
            const center = left + delta;
            const right = center + delta;
            for (0..n_bins - 1) |i| {
                const f = cbxKaldiMel(bin_hz * @as(f64, @floatFromInt(i)));
                if (f > left and f < right) {
                    const w = if (f <= center) (f - left) / (center - left) else (right - f) / (right - center);
                    filters.data.items[m * n_bins + i] = @floatCast(w);
                }
            }
        }
    }

    // s3 tokenizer: librosa slaney mel, 128 bands
    try self.s3tok_cache.fillSinCosTable(400);
    try self.s3tok_cache.fillHannWindow(400, true);
    try self.s3tok_cache.fillMelFilterbankMatrix(128, 400, 16000, 0.0, -1.0, true, 1.0, false);

    // voice encoder: slaney mel, 40 bands up to 8 kHz
    try self.ve_cache.fillSinCosTable(400);
    try self.ve_cache.fillHannWindow(400, true);
    try self.ve_cache.fillMelFilterbankMatrix(40, 400, 16000, 0.0, 8000.0, true, 1.0, false);

    // s3gen prompt features at the 24 kHz decoder rate
    try self.matcha_cache.fillSinCosTable(1920);
    try self.matcha_cache.fillHannWindow(1920, true);
    try self.matcha_cache.fillMelFilterbankMatrix(80, 1920, 24000, 0.0, 8000.0, true, 1.0, false);
}

/// `clip.resize((clip.size() + 639) / 640 * 640, 0.0f)` over the first
/// `cap` samples of `pcm`: the clip padded to whole 40 ms tokens. The caller
/// owns the list.
fn cbxCapClip(pcm: []const f32, cap: usize) !std.ArrayList(f32) {
    var clip: std.ArrayList(f32) = .empty;
    errdefer clip.deinit(gpa);
    try clip.appendSlice(gpa, pcm[0..@min(pcm.len, cap)]);
    try clip.appendNTimes(gpa, 0.0, (clip.items.len + 639) / 640 * 640 - clip.items.len);
    return clip;
}

/// A lambda of `mtmd_audio_preprocessor_chatterbox_ref`'s `preprocess`,
/// its `s3tok_entry` (mtmd-audio.cpp:1311 @b1b770d90).
fn cbxS3tokEntry(self: *const Preprocessor, pcm: []const f32, cap: usize, output: *std.ArrayList(Mel)) !bool {
    var clip = try cbxCapClip(pcm, cap);
    defer clip.deinit(gpa);
    var mel: std.ArrayList(f32) = .empty;
    errdefer mel.deinit(gpa);
    var n_frames: c_int = 0;
    if (!try cbxS3tokLogMel(&self.s3tok_cache, clip.items, &mel, &n_frames)) {
        mel.deinit(gpa);
        return false;
    }
    try output.append(gpa, .{ .n_len = n_frames, .n_len_org = n_frames, .n_mel = self.s3tok_cache.filters.n_mel, .data = mel });
    return true;
}

/// Ports `process` (mtmd-audio.cpp:1233 @b1b770d90), a member of
/// `mtmd_audio_preprocessor_chatterbox_ref`: one clip to five entries --
/// the CAMPPlus kaldi fbank, the s3 tokenizer log-mels at the flow and t3
/// caps, the s3gen prompt features and the voice encoder's power mel.
fn cbxRefPreprocess(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel)) !bool {
    for (output.items) |*m| m.deinit();
    output.clearRetainingCapacity();
    const hp = self.hparams;
    const sr: c_int = hp.audio_sample_rate;

    // the turbo variant loudness-normalizes the whole clip before any feature
    const pcm = try gpa.dupe(f32, samples);
    defer gpa.free(pcm);
    if (!self.is_mtl) {
        const lufs = try cbxLufs(pcm, sr);
        if (lufs != -std.math.inf(f32)) {
            const gain = pow10f((-27.0 - lufs) / 20.0);
            if (std.math.isFinite(gain) and gain > 0.0) {
                for (pcm) |*v| v.* *= gain;
            }
        }
    }

    // entry 0: torchaudio kaldi.fbank of CAMPPlus (snip edges, dc removal, preemphasis 0.97), mean subtracted over time
    {
        const frame_len: c_int = 400;
        const hop: c_int = 160;
        const n_fft: c_int = 512;
        const n_mel: c_int = hp.n_mel_bins;

        if (@as(c_int, @intCast(pcm.len)) < frame_len) {
            return false;
        }
        const n_frames: c_int = 1 + @divTrunc(@as(c_int, @intCast(pcm.len)) - frame_len, hop);
        const nf: usize = @intCast(n_frames);
        const nm: usize = @intCast(n_mel);

        var out: Mel = .{ .n_len = n_frames, .n_len_org = n_frames, .n_mel = n_mel };
        errdefer out.deinit();
        try out.data.appendNTimes(gpa, 0.0, nm * nf);

        const win = self.cache.hann_window.items;
        var bufs = try CbxFftBufs.init(n_fft);
        defer bufs.deinit();
        var e: std.ArrayList(f64) = .empty;
        defer e.deinit(gpa);
        const fl: usize = @intCast(frame_len);
        for (0..nf) |fr| {
            const x = pcm[fr * @as(usize, @intCast(hop)) ..][0..fl];

            var mean: f64 = 0.0;
            for (x) |v| mean += v;
            mean /= @as(f64, frame_len);

            @memset(bufs.in, 0.0);
            bufs.in[0] = @floatCast((@as(f64, x[0]) - mean) * (@as(f64, 1.0) - 0.97) * @as(f64, win[0]));
            for (1..fl) |i| {
                bufs.in[i] = @floatCast(@mulAdd(f64, -0.97, @as(f64, x[i - 1]) - mean, @as(f64, x[i]) - mean) * @as(f64, win[i]));
            }
            try cbxMelEnergies(&self.cache, bufs.in, bufs.out, false, &e);
            for (0..nm) |m| {
                out.data.items[m * nf + fr] = @floatCast(libm.log(maxF64(e.items[m], @as(f64, std.math.floatEps(f32)))));
            }
        }

        for (0..nm) |m| {
            const row = out.data.items[m * nf ..][0..nf];
            var mean: f64 = 0.0;
            for (row) |v| mean += v;
            mean /= @as(f64, @floatFromInt(n_frames));
            for (row) |*v| v.* -= @as(f32, @floatCast(mean));
        }

        try output.append(gpa, out);
    }

    // entries 1 and 2: s3 tokenizer log-mels at the flow and t3 caps, padded to whole 40 ms tokens
    const sr_u: usize = @intCast(sr);
    const gen_cap: usize = 10 * sr_u;
    const t3_cap: usize = @as(usize, if (self.is_mtl) 6 else 15) * sr_u;
    if (!try cbxS3tokEntry(self, pcm, gen_cap, output) or !try cbxS3tokEntry(self, pcm, t3_cap, output)) {
        return false;
    }

    // entry 3: s3gen prompt features of the flow-capped clip at 24 kHz
    {
        var clip = try cbxCapClip(pcm, gen_cap);
        defer clip.deinit(gpa);
        var pcm24: std.ArrayList(f32) = .empty;
        defer pcm24.deinit(gpa);
        try cbxUpsample32(clip.items, &pcm24);
        var feat: std.ArrayList(f32) = .empty;
        errdefer feat.deinit(gpa);
        var n_frames: c_int = 0;
        if (!try cbxMatchaLogMel(&self.matcha_cache, pcm24.items, &feat, &n_frames)) {
            feat.deinit(gpa);
            return false;
        }
        try output.append(gpa, .{ .n_len = n_frames, .n_len_org = n_frames, .n_mel = 80, .data = feat });
    }

    // entry 4: voice encoder mel of the silence-trimmed clip, padded to the 160-frame partial grid (rate 1.3)
    {
        var t0: usize = 0;
        var t1: usize = 0;
        try cbxTrimSilence(pcm, 20.0, &t0, &t1);
        if (t1 <= t0) {
            return false;
        }
        var mel: std.ArrayList(f32) = .empty;
        errdefer mel.deinit(gpa);
        var n_frames: c_int = 0;
        if (!try cbxVeMel(&self.ve_cache, pcm[t0..t1], &mel, &n_frames)) {
            mel.deinit(gpa);
            return false;
        }
        const n_mel: c_int = 40;
        const n_partial: c_int = 160;
        const step: c_int = comptime @intFromFloat(@round((@as(f64, 16000.0) / 1.3) / @as(f64, n_partial)));
        const span = @max(n_frames - n_partial + step, 0);
        var n_wins: c_int = @divTrunc(span, step);
        const rem: c_int = @rem(span, step);
        if (n_wins == 0 or @as(f64, @floatFromInt(rem + n_partial - step)) / @as(f64, n_partial) >= 0.8) {
            n_wins += 1;
        }
        const target: c_int = n_partial + step * (n_wins - 1);
        const new_len: usize = @as(usize, @intCast(target)) * @as(usize, @intCast(n_mel));
        if (new_len > mel.items.len) {
            try mel.appendNTimes(gpa, 0.0, new_len - mel.items.len);
        } else {
            mel.shrinkRetainingCapacity(new_len);
        }
        try output.append(gpa, .{ .n_len = target, .n_len_org = target, .n_mel = n_mel, .data = mel });
    }
    return true;
}

/// Ports `mtmd_audio_preproc_out` (mtmd-audio.h:78 @b1b770d90): one entry
/// per chunk (empty on failure), and the input of `compute` when the entries
/// do not hold the data yet.
pub const PreprocOut = struct {
    entries: std.ArrayList(impl.ImageF32) = .empty,
    samples: std.ArrayList(f32) = .empty,

    pub fn deinit(self: *PreprocOut) void {
        for (self.entries.items) |*e| e.deinit();
        self.entries.deinit(gpa);
        self.samples.deinit(gpa);
    }

    /// Ports `needs_compute` (mtmd-audio.h:82 @b1b770d90), a member of `mtmd_audio_preproc_out`.
    pub fn needsCompute(self: *const PreprocOut) bool {
        return self.samples.items.len != 0 and (self.entries.items.len == 0 or self.entries.items[0].isPlaceholder());
    }
};

fn freeMels(mels: *std.ArrayList(Mel)) void {
    for (mels.items) |*m| m.deinit();
    mels.deinit(gpa);
}

/// Ports `mtmd_audio_fill_entries` (mtmd-audio.cpp:273 @b1b770d90): one
/// audio entry per mel, a placeholder when `shape_only` or the mel has no
/// data. Replaces what `entries` held.
fn fillEntries(mels: []const Mel, shape_only: bool, entries: *std.ArrayList(impl.ImageF32)) !void {
    for (entries.items) |*e| e.deinit();
    entries.clearRetainingCapacity();
    for (mels) |*mel| {
        impl.ggmlAssert(mel.n_len <= std.math.maxInt(i32) and mel.n_len >= 0, "mtmd-audio.cpp", 276, "mel.n_len <= INT32_MAX && mel.n_len >= 0");
        impl.ggmlAssert(mel.n_mel <= std.math.maxInt(i32) and mel.n_mel >= 0, "mtmd-audio.cpp", 277, "mel.n_mel <= INT32_MAX && mel.n_mel >= 0");
        var entry: impl.ImageF32 = .{};
        errdefer entry.deinit();
        try entry.setSize(.{ .width = @intCast(mel.n_len), .height = @intCast(mel.n_mel) }, shape_only or mel.data.items.len == 0, true);
        if (!entry.isPlaceholder()) {
            try entry.cpyBuf(mel.data.items);
        }
        try entries.append(gpa, entry);
    }
}

/// The sixteen subclasses of `mtmd_audio_preprocessor`
/// (mtmd-audio.h:87 @b1b770d90), in the header's order -- later ones
/// (`d1omni`, `nemotron_transcribe` and `chatterbox_ref`)
/// last, so the earlier kinds keep the ordinals `harness/mtmd_audio_ref.cpp`
/// switches on.
pub const Kind = enum {
    whisper,
    conformer,
    granite_speech,
    gemma4a,
    gemma4ua,
    qwen3a,
    dots3note,
    mimo_audio,
    qwen3tts_spk,
    pockettts,
    parakeet,
    /// `mtmd_audio_preprocessor_d1omni` (mtmd-audio.h:141 @b1b770d90): the
    /// conformer's, on audio cut to 30 s and padded to 0.5 s
    d1omni,
    /// `mtmd_audio_preprocessor_lfm3a` (mtmd-audio.h:146 @b1b770d90)
    lfm3a,
    /// `mtmd_audio_preprocessor_muse_spark_audio` (mtmd-audio.h:124 @b1b770d90)
    muse_spark_audio,
    /// `mtmd_audio_preprocessor_nemotron_transcribe` (mtmd-audio.h:258
    /// @b1b770d90): parakeet's `initialize` and log-mel, one raw chunk
    nemotron_transcribe,
    /// `mtmd_audio_preprocessor_chatterbox_ref` (mtmd-audio.h:232
    /// @b1b770d90): five entries per clip, `Preprocessor.is_mtl` picks the
    /// variant
    chatterbox_ref,
};

/// Ports `mtmd_audio_preprocessor` (mtmd-audio.h:87 @b1b770d90) and its
/// subclasses: hparams it borrows from the model, the log-mel config
/// `initialize` fills, and the cache the subclass owns.
pub const Preprocessor = struct {
    kind: Kind,
    hparams: *const Hparams,
    /// filled by `initialize` of the preprocessors that use the log-mel
    /// graph; the owner sets `n_threads` and `sched` before `initialize`
    mel_config: MelConfig = .{},
    cache: Cache = .{},
    /// chatterbox_ref only: the multilingual variant (no loudness
    /// normalization, 6 s t3 cap); `cache` is its `fbank_cache`
    is_mtl: bool = false,
    s3tok_cache: Cache = .{},
    matcha_cache: Cache = .{},
    ve_cache: Cache = .{},

    /// `segment_seconds` of `mtmd_audio_preprocessor_gemma4a`
    /// (mtmd-audio.h:164 @b1b770d90): each segment is padded and encoded on
    /// its own
    pub const gemma4a_segment_seconds: c_int = 30;

    /// Return: a preprocessor reading `hparams`, which must outlive it.
    /// `initialize` must run before `preprocess`, and `mel_config.sched`
    /// must be set before the first `compute`.
    pub fn init(kind: Kind, hparams: *const Hparams) Preprocessor {
        return .{ .kind = kind, .hparams = hparams };
    }

    pub fn deinit(self: *Preprocessor) void {
        self.cache.deinit();
        self.s3tok_cache.deinit();
        self.matcha_cache.deinit();
        self.ve_cache.deinit();
    }

    /// Ports each subclass's `initialize`, not thread-safe. The C++'s
    /// per-class bodies, in its order:
    ///
    /// - `initialize` of whisper (mtmd-audio.cpp:504 @b1b770d90)
    /// - `initialize` of qwen3a (mtmd-audio.cpp:591 @b1b770d90)
    /// - `initialize` of dots3note (mtmd-audio.cpp:694 @b1b770d90)
    /// - `initialize` of mimo-audio (mtmd-audio.cpp:789 @b1b770d90)
    /// - `initialize` of qwen3tts-spk (mtmd-audio.cpp:855 @b1b770d90)
    /// - `initialize` of the conformer (mtmd-audio.cpp:1388 @b1b770d90)
    /// - `initialize` of lfm3a (mtmd-audio.cpp:1473 @b1b770d90)
    /// - `initialize` of granite-speech (mtmd-audio.cpp:1529 @b1b770d90)
    /// - `initialize` of gemma4a (mtmd-audio.cpp:1633 @b1b770d90)
    /// - `initialize` of parakeet (mtmd-audio.cpp:1715 @b1b770d90)
    /// - `initialize` of gemma4ua (mtmd-audio.cpp:1849 @b1b770d90)
    pub fn initialize(self: *Preprocessor) !void {
        const hp = self.hparams;
        const n_fft: u32 = @bitCast(hp.audio_n_fft);
        switch (self.kind) {
            .whisper, .qwen3a, .dots3note, .qwen3tts_spk, .lfm3a => {
                try self.cache.fillSinCosTable(n_fft);
                try self.cache.fillHannWindow(@bitCast(hp.audio_window_len), true);
                try self.cache.fillMelFilterbankMatrix(hp.n_mel_bins, hp.audio_n_fft, hp.audio_sample_rate, 0.0, -1.0, true, 1.0, false);
            },
            .conformer, .d1omni => {
                try self.cache.fillSinCosTable(n_fft);
                // NeMo uses a symmetric window: torch.hann_window(periodic=False)
                try self.cache.fillHannWindow(@bitCast(hp.audio_window_len), false);
                try self.cache.fillMelFilterbankMatrix(hp.n_mel_bins, hp.audio_n_fft, hp.audio_sample_rate, 0.0, -1.0, true, 1.0, false);
            },
            .mimo_audio => {
                try self.cache.fillSinCosTable(n_fft);
                try self.cache.fillHannWindow(@bitCast(hp.audio_window_len), true);
                try self.cache.fillMelFilterbankMatrix(hp.n_mel_bins, hp.audio_n_fft, hp.audio_sample_rate, 0.0, @as(f32, @floatFromInt(hp.audio_sample_rate)) / 2.0, false, 1.0, true);
            },
            .granite_speech => {
                try self.cache.fillSinCosTable(n_fft);
                try self.cache.fillHannWindow(@bitCast(hp.audio_window_len), true);
                try self.cache.fillMelFilterbankMatrix(@divTrunc(hp.n_mel_bins, 2), hp.audio_n_fft, hp.audio_sample_rate, 0.0, -1.0, false, 1.0, true);
            },
            .gemma4a => {
                try self.cache.fillSinCosTable(n_fft);

                // Standard periodic Hann window, zero-padded to FFT size
                self.cache.hann_window.clearRetainingCapacity();
                try self.cache.hann_window.appendNTimes(gpa, 0, @intCast(hp.audio_n_fft));
                var i: u32 = 0;
                while (i < @as(u32, @bitCast(hp.audio_window_len))) : (i += 1) {
                    const x = (2.0 * @as(f32, @floatCast(m_pi)) * @as(f32, @floatFromInt(i))) / @as(f32, @floatFromInt(hp.audio_window_len));
                    self.cache.hann_window.items[i] = @mulAdd(f32, -0.5, libm.cosf(x), 0.5);
                }

                // HTK mel scale, no Slaney area normalization
                try self.cache.fillMelFilterbankMatrix(hp.n_mel_bins, hp.audio_n_fft, hp.audio_sample_rate, 0.0, @as(f32, @floatFromInt(hp.audio_sample_rate)) / 2.0, false, 1.0, true);
            },
            .parakeet, .nemotron_transcribe => {
                try self.cache.fillSinCosTable(n_fft);

                const n_fft_bins: usize = @intCast(@divTrunc(hp.audio_n_fft, 2) + 1);
                ggmlAssert(hp.mel_filters.items.len == @as(usize, @intCast(hp.n_mel_bins)) * n_fft_bins, 1719, "hparams.mel_filters.size() == (size_t)hparams.n_mel_bins * n_fft");
                self.cache.filters.n_mel = hp.n_mel_bins;
                self.cache.filters.n_fft = @intCast(n_fft_bins);
                self.cache.filters.data.clearRetainingCapacity();
                try self.cache.filters.data.appendSlice(gpa, hp.mel_filters.items);

                ggmlAssert(hp.window.items.len == @as(usize, @intCast(hp.audio_window_len)), 1724, "hparams.window.size() == (size_t)hparams.audio_window_len");
                ggmlAssert(hp.window.items.len <= @as(usize, @intCast(hp.audio_n_fft)), 1725, "hparams.window.size() <= (size_t) hparams.audio_n_fft");
                self.cache.hann_window.clearRetainingCapacity();
                try self.cache.hann_window.appendSlice(gpa, hp.window.items);
            },
            .gemma4ua, .pockettts, .muse_spark_audio => {},
            .chatterbox_ref => try cbxRefInitialize(self),
        }

        // the log-mel config, as each subclass's `initialize` fills it
        const mc = &self.mel_config;
        switch (self.kind) {
            .gemma4ua, .pockettts, .muse_spark_audio, .chatterbox_ref => return,
            else => {},
        }
        mc.n_mel = hp.n_mel_bins;
        mc.n_fft_bins = 1 + @divTrunc(hp.audio_n_fft, 2);
        mc.hann_window_size = hp.audio_window_len;
        mc.hop_length = hp.audio_hop_len;
        mc.sample_rate = hp.audio_sample_rate;
        mc.has_cache = true;
        switch (self.kind) {
            .whisper => {
                mc.center_padding = false;
                mc.preemph = 0.0; // disabled
                mc.use_natural_log = false;
                mc.norm_per_feature = false;
            },
            .qwen3a => {
                mc.no_padding = true; // reflection padding applied in preprocess()
                mc.use_natural_log = false; // log10
            },
            .dots3note => {
                mc.no_padding = true; // padding applied in preprocess()
                mc.use_natural_log = false;
            },
            .mimo_audio => {
                mc.no_padding = true; // reflect padding applied in preprocess()
                mc.use_natural_log = true;
                mc.use_magnitude = true;
                mc.mel_floor = 1e-7;
                mc.norm_per_feature = false;
            },
            .qwen3tts_spk => {
                mc.no_padding = true; // reflect padding applied in preprocess()
                mc.use_natural_log = true;
                mc.use_magnitude = true;
                mc.mel_floor = 1e-5;
            },
            .conformer, .d1omni => {
                mc.center_padding = true;
                mc.preemph = 0.97;
                mc.use_natural_log = true;
                mc.norm_per_feature = true;
                mc.mel_floor_add = true;
                mc.std_eps_after_sqrt = true;
            },
            .lfm3a => {
                mc.no_padding = true;
                mc.mel_floor = 1e-12; // log10 floor of -12
            },
            .granite_speech => {
                mc.n_mel = @divTrunc(hp.n_mel_bins, 2);
                mc.no_padding = true;
                mc.center_padding = false;
                mc.preemph = 0.0;
                mc.use_natural_log = false;
                mc.norm_per_feature = false;
                mc.mel_floor = 1e-10;
            },
            .gemma4a => {
                mc.hann_window_size = hp.audio_n_fft; // window is zero-padded to FFT size
                mc.no_padding = true;
                mc.center_padding = false;
                mc.preemph = 0.0;
                mc.use_natural_log = true;
                mc.use_magnitude = true;
                mc.mel_floor = 0.001;
                mc.norm_per_feature = false;
            },
            .parakeet, .nemotron_transcribe => {
                mc.no_padding = true; // preemphasis and padding already applied in log_mel()
                mc.use_natural_log = true;
                mc.mel_floor_add = true;
            },
            .gemma4ua, .pockettts, .muse_spark_audio, .chatterbox_ref => unreachable,
        }
    }

    /// Ports `preprocess` (mtmd-audio.cpp:287 @b1b770d90), a member of
    /// `mtmd_audio_preprocessor`: the entries with their final shapes, and no
    /// data -- the samples kept for `compute` -- when the preprocessor uses
    /// the log-mel graph. Empty entries on failure. The caller owns the
    /// result.
    pub fn preprocess(self: *const Preprocessor, samples: []const f32) !PreprocOut {
        var out: PreprocOut = .{};
        errdefer out.deinit();
        const deferred = self.mel_config.has_cache;
        var mels: std.ArrayList(Mel) = .empty;
        defer freeMels(&mels);
        if (try self.process(samples, &mels, deferred)) {
            try fillEntries(mels.items, deferred, &out.entries);
            if (deferred) {
                try out.samples.appendSlice(gpa, samples);
            }
        }
        return out;
    }

    /// Ports `compute` (mtmd-audio.cpp:300 @b1b770d90), a member of
    /// `mtmd_audio_preprocessor`: runs the log-mel graph `preprocess`
    /// deferred, on `mel_config.sched`. Run it on the thread that encodes.
    ///
    /// Return: false when the preprocessor fails, `out` then unchanged.
    pub fn compute(self: *const Preprocessor, out: *PreprocOut) !bool {
        if (!out.needsCompute()) {
            return true;
        }
        var mels: std.ArrayList(Mel) = .empty;
        defer freeMels(&mels);
        if (!try self.process(out.samples.items, &mels, false)) {
            return false;
        }
        try fillEntries(mels.items, false, &out.entries);
        out.samples.clearAndFree(gpa);
        return true;
    }

    /// Ports each subclass's `process`: `samples` to the mels appended to
    /// `output`, which the caller owns. `shape_only` skips the log-mel graph:
    /// the shapes are right, the data is zero-filled.
    ///
    /// Return: whether it succeeded, as the C++ returns it.
    pub fn process(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        return switch (self.kind) {
            .whisper => self.whisper(samples, output, shape_only),
            .qwen3a => self.qwen3a(samples, output, shape_only),
            .dots3note => self.dots3note(samples, output, shape_only),
            .mimo_audio => self.mimoAudio(samples, output, shape_only),
            .qwen3tts_spk => self.qwen3ttsSpk(samples, output, shape_only),
            .conformer => self.conformer(samples, output, shape_only),
            .d1omni => self.d1omni(samples, output, shape_only),
            .granite_speech => self.graniteSpeech(samples, output, shape_only),
            .gemma4a => self.gemma4a(samples, output, shape_only),
            .parakeet => self.parakeet(samples, output, shape_only),
            .gemma4ua => self.gemma4ua(samples, output),
            .pockettts => self.pockettts(samples, output),
            .lfm3a => self.lfm3a(samples, output, shape_only),
            .muse_spark_audio => self.museSparkAudio(samples, output),
            .nemotron_transcribe => self.nemotronTranscribe(samples, output, shape_only),
            .chatterbox_ref => cbxRefPreprocess(self, samples, output),
        };
    }

    fn assertCache(self: *const Preprocessor, comptime line: c_int) void {
        ggmlAssert(self.cache.sin_vals.items.len != 0, line, "!cache.sin_vals.empty()");
        ggmlAssert(self.cache.cos_vals.items.len != 0, line + 1, "!cache.cos_vals.empty()");
        ggmlAssert(self.cache.filters.data.items.len != 0, line + 2, "!cache.filters.data.empty()");
    }

    /// Reflection padding by `pad` at each end, zero where the source runs
    /// out, as qwen3a, mimo-audio and dots3note write it.
    fn reflectPad(samples: []const f32, pad: c_int, pad_end: bool) !std.ArrayList(f32) {
        const n: c_int = @intCast(samples.len);
        const pu: usize = @intCast(pad);
        var padded: std.ArrayList(f32) = .empty;
        try padded.appendNTimes(gpa, 0, samples.len + 2 * pu);
        var i: c_int = 0;
        while (i < pad) : (i += 1) {
            const src = pad - i;
            padded.items[@intCast(i)] = if (src < n) samples[@intCast(src)] else 0.0;
        }
        @memcpy(padded.items[pu..][0..samples.len], samples);
        if (pad_end) {
            i = 0;
            while (i < pad) : (i += 1) {
                const src = n - 2 - i;
                padded.items[samples.len + pu + @as(usize, @intCast(i))] = if (src >= 0) samples[@intCast(src)] else 0.0;
            }
        }
        return padded;
    }

    /// Ports `process` (mtmd-audio.cpp:521 @b1b770d90), a member of `mtmd_audio_preprocessor_whisper`.
    fn whisper(self: *const Preprocessor, samples_in: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        const hp = self.hparams;
        if (samples_in.len == 0) return false; // empty audio

        var samples = samples_in;
        var smpl: std.ArrayList(f32) = .empty;
        defer smpl.deinit(gpa);
        // reflection padding needs one sample plus half an FFT window
        const min_samples: usize = @as(usize, @intCast(hp.audio_n_fft)) / 2 + 1;
        if (samples.len < min_samples) {
            try smpl.appendNTimes(gpa, 0, min_samples);
            @memcpy(smpl.items[0..samples.len], samples);
            samples = smpl.items;
        }

        // make sure the cache is initialized
        self.assertCache(541);

        var out_full: Mel = .{};
        defer out_full.deinit();
        if (!try logMelSpectrogram(samples, &self.mel_config, &self.cache, shape_only, &out_full)) return false;

        // the cgraph only accepts 3000 frames each, so the mel is split; it
        // always has 3000 silent frames at the end
        const frames_per_chunk: usize = 3000;
        ggmlAssert(@as(usize, @intCast(out_full.n_len)) > frames_per_chunk, 557, "(size_t) out_full.n_len > frames_per_chunk");
        const full_len: usize = @intCast(out_full.n_len);
        var off: usize = 0;
        while (off < full_len) : (off += frames_per_chunk) {
            const n_len = @min(@as(i64, frames_per_chunk), out_full.n_len - @as(i64, @intCast(off)));
            if (n_len < frames_per_chunk) break; // last incomplete chunk will always be a padded chunk, safe to ignore

            var out_chunk: Mel = .{ .n_len = n_len, .n_mel = out_full.n_mel, .n_len_org = out_full.n_mel }; // n_len_org unused
            try out_chunk.data.ensureTotalCapacity(gpa, @as(usize, @intCast(out_chunk.n_mel)) * @as(usize, @intCast(out_chunk.n_len)));
            var i: usize = 0;
            while (i < out_full.n_mel) : (i += 1) {
                const src = out_full.data.items[i * full_len + off ..][0..frames_per_chunk];
                out_chunk.data.appendSliceAssumeCapacity(src);
            }
            try output.append(gpa, out_chunk);
        }
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:606 @b1b770d90), a member of `mtmd_audio_preprocessor_qwen3a`:
    /// the WhisperFeatureExtractor with `truncation=False`.
    fn qwen3a(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        const hp = self.hparams;
        if (samples.len == 0) return false;
        self.assertCache(614);

        // Reflection-pad n_fft/2 samples at each end, matching WhisperFeatureExtractor center=True
        const pad = @divTrunc(hp.audio_n_fft, 2);
        var padded = try reflectPad(samples, pad, true);
        defer padded.deinit(gpa);

        var mel_full: Mel = .{};
        defer mel_full.deinit();
        if (!try logMelSpectrogram(padded.items, &self.mel_config, &self.cache, shape_only, &mel_full)) return false;

        // Whisper-style normalization: clamp to (max - 8), scale to [-1, 1]
        {
            var mmax: f64 = -1e20;
            for (mel_full.data.items) |v| {
                if (v > mmax) mmax = v;
            }
            mmax -= 8.0;
            for (mel_full.data.items) |*v| v.* = @floatCast((maxF64(v.*, mmax) + 4.0) / 4.0);
        }

        // the center-padded STFT gives ~n_samples/hop_length frames
        const n_eff = @min(mel_full.n_len, @as(i64, @intCast(samples.len / @as(usize, @intCast(hp.audio_hop_len)))) + 1);

        // inference windows of n_window_infer=800, each padded to the next
        // multiple of the conv sub-chunk size for the cgraph
        const chunk_size: i64 = 100; // conv sub-chunk size (n_window * 2, n_window=50)
        const window_size: i64 = 800; // mel frames per forward pass (n_window_infer=800)

        var off: i64 = 0;
        while (off < n_eff) : (off += window_size) {
            const win_eff = @min(window_size, n_eff - off);
            const n_chunks = @divTrunc(win_eff + chunk_size - 1, chunk_size);
            const n_padded = n_chunks * chunk_size;

            var out: Mel = .{ .n_mel = mel_full.n_mel, .n_len = n_padded, .n_len_org = win_eff };
            try out.data.appendNTimes(gpa, 0, @as(usize, @intCast(out.n_mel)) * @as(usize, @intCast(out.n_len)));
            var m: i64 = 0;
            while (m < out.n_mel) : (m += 1) {
                const copy_len = @min(win_eff, mel_full.n_len - off);
                if (copy_len > 0) {
                    const src = mel_full.data.items[@intCast(m * mel_full.n_len + off)..][0..@intCast(copy_len)];
                    @memcpy(out.data.items[@intCast(m * out.n_len)..][0..@intCast(copy_len)], src);
                }
            }
            try output.append(gpa, out);
        }
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:709 @b1b770d90), a member of `mtmd_audio_preprocessor_dots3note`:
    /// whisper-style log-mel per 60 s chunk, only the valid frames kept.
    fn dots3note(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        const hp = self.hparams;
        if (samples.len == 0) return false;
        self.assertCache(717);

        const pad = @divTrunc(hp.audio_n_fft, 2); // center=True padding
        const hop = hp.audio_hop_len;
        const chunk_samples: usize = @as(usize, @intCast(hp.audio_chunk_len)) *% @as(usize, @intCast(hp.audio_sample_rate));

        var start: usize = 0;
        while (start < samples.len) : (start += chunk_samples) {
            const n_chunk = @min(chunk_samples, samples.len - start);
            const chunk = samples[start..][0..n_chunk];

            const n_valid: i64 = @intCast(n_chunk / @as(usize, @intCast(hop)));
            if (n_valid == 0) continue; // sub-hop tail, contributes no frames

            // reflect-pad the start; a partial chunk sees zeros past its end,
            // a full chunk reflects its own tail
            var padded = try reflectPad(chunk, pad, n_chunk == chunk_samples);
            defer padded.deinit(gpa);

            var mel_full: Mel = .{};
            defer mel_full.deinit();
            if (!try logMelSpectrogram(padded.items, &self.mel_config, &self.cache, shape_only, &mel_full)) return false;
            ggmlAssert(mel_full.n_len >= n_valid, 753, "mel_full.n_len >= n_valid");

            // per-chunk whisper-style normalization, then keep only the valid frames
            var out: Mel = .{ .n_mel = mel_full.n_mel, .n_len = n_valid, .n_len_org = n_valid };
            try out.data.resize(gpa, @as(usize, @intCast(out.n_mel)) * @as(usize, @intCast(out.n_len)));

            var mmax: f64 = -1e20;
            var m: i64 = 0;
            while (m < out.n_mel) : (m += 1) {
                var t: i64 = 0;
                while (t < n_valid) : (t += 1) mmax = maxF64(mmax, mel_full.data.items[@intCast(m * mel_full.n_len + t)]);
            }
            mmax -= 8.0;
            m = 0;
            while (m < out.n_mel) : (m += 1) {
                var t: i64 = 0;
                while (t < n_valid) : (t += 1) {
                    const v = maxF64(mel_full.data.items[@intCast(m * mel_full.n_len + t)], mmax);
                    out.data.items[@intCast(m * n_valid + t)] = @floatCast((v + 4.0) / 4.0);
                }
            }

            try output.append(gpa, out);
        }
        return output.items.len != 0;
    }

    /// Ports `process` (mtmd-audio.cpp:813 @b1b770d90), a member of `mtmd_audio_preprocessor_mimo_audio`:
    /// torchaudio's `MelSpectrogram(power=1.0, center=True)` then `log(clip(spec, 1e-7))`.
    fn mimoAudio(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        const hp = self.hparams;
        if (samples.len == 0) return false;
        self.assertCache(821);

        const pad = @divTrunc(hp.audio_n_fft, 2);
        var padded = try reflectPad(samples, pad, true);
        defer padded.deinit(gpa);

        var out: Mel = .{};
        if (!try logMelSpectrogram(padded.items, &self.mel_config, &self.cache, shape_only, &out)) {
            out.deinit();
            return false;
        }
        try output.append(gpa, out);
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:872 @b1b770d90), a member of `mtmd_audio_preprocessor_qwen3tts_spk`:
    /// `mel_spectrogram()` of `modeling_qwen3_tts.py`, the whole clip at once.
    fn qwen3ttsSpk(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        const hp = self.hparams;
        if (samples.len == 0) return false;
        self.assertCache(880);

        // reflect pad by (n_fft - hop) / 2, matching center=False STFT framing
        const pad = @divTrunc(hp.audio_n_fft - hp.audio_hop_len, 2);
        if (samples.len < @as(usize, @intCast(pad)) + 1) return false;

        const pu: usize = @intCast(pad);
        const n = samples.len;
        var padded: std.ArrayList(f32) = .empty;
        defer padded.deinit(gpa);
        try padded.appendNTimes(gpa, 0, n + 2 * pu);
        for (0..pu) |i| padded.items[i] = samples[pu - i];
        @memcpy(padded.items[pu..][0..n], samples);
        for (0..pu) |i| padded.items[n + pu + i] = samples[n - 2 - i];

        var out: Mel = .{};
        if (!try logMelSpectrogram(padded.items, &self.mel_config, &self.cache, shape_only, &out)) {
            out.deinit();
            return false;
        }
        try output.append(gpa, out);
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:1408 @b1b770d90), a member of `mtmd_audio_preprocessor_conformer`.
    fn conformer(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        // empty audio
        if (samples.len == 0) return false;

        // make sure the cache is initialized
        self.assertCache(1418);

        var out_full: Mel = .{};
        if (!try logMelSpectrogram(samples, &self.mel_config, &self.cache, shape_only, &out_full)) {
            out_full.deinit();
            return false;
        }
        try output.append(gpa, out_full);
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:1436 @b1b770d90), a member of
    /// `mtmd_audio_preprocessor_d1omni`: the conformer's, on the audio cut to
    /// 30 s and padded to 0.5 s (d1-omni audio.py), each mel then cut to one
    /// frame per hop.
    fn d1omni(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        const hp = self.hparams;
        if (samples.len == 0) return false;
        const n_max: usize = @intCast(30 * hp.audio_sample_rate);
        const n_min: usize = @intCast(@divTrunc(hp.audio_sample_rate, 2));

        var buf: std.ArrayList(f32) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, samples[0..@min(samples.len, n_max)]);
        if (buf.items.len < n_min) try buf.appendNTimes(gpa, 0.0, n_min - buf.items.len);

        if (!try self.conformer(buf.items, output, shape_only)) return false;

        // the encoder reads one frame per hop, not the extra frame of the centre padding (NeMo: seq_len)
        const n_frames: i64 = @intCast(buf.items.len / @as(usize, @bitCast(@as(isize, hp.audio_hop_len))));
        for (output.items) |*mel| {
            if (mel.n_len <= n_frames) continue;
            const nf: usize = @intCast(n_frames);
            const n_len: usize = @intCast(mel.n_len);
            const data = try gpa.alloc(f32, @as(usize, @intCast(mel.n_mel)) * nf);
            var j: usize = 0;
            while (j < @as(usize, @intCast(mel.n_mel))) : (j += 1) {
                @memcpy(data[j * nf ..][0..nf], mel.data.items[j * n_len ..][0..nf]);
            }
            mel.n_len = n_frames;
            mel.data.deinit(gpa);
            mel.data = .fromOwnedSlice(data);
        }
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:1488 @b1b770d90), a member of
    /// `mtmd_audio_preprocessor_lfm3a`.
    fn lfm3a(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        const hp = self.hparams;
        if (samples.len == 0) return false;

        const win: usize = @intCast(hp.audio_window_len);
        const hop: usize = @intCast(hp.audio_hop_len);

        // pad on the right so that no sample is dropped by the last frame
        var n_padded = win;
        if (samples.len > win) n_padded = samples.len + (hop - (samples.len - win) % hop) % hop;
        var padded: std.ArrayList(f32) = .empty;
        defer padded.deinit(gpa);
        try padded.appendSlice(gpa, samples);
        try padded.appendNTimes(gpa, 0.0, n_padded - samples.len);

        self.assertCache(1507);

        var out: Mel = .{};
        if (!try logMelSpectrogram(padded.items, &self.mel_config, &self.cache, shape_only, &out)) {
            out.deinit();
            return false;
        }

        for (out.data.items) |*v| v.* = (v.* + 4.0) / 4.0;

        try output.append(gpa, out);
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:2028 @b1b770d90), a member of
    /// `mtmd_audio_preprocessor_muse_spark_audio`: Muse Spark reads the raw
    /// waveform, so the samples pass through, zero padded to a whole number of
    /// embeddings.
    fn museSparkAudio(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel)) !bool {
        const fold: i64 = self.hparams.audioSamplesPerToken();
        const n_samples: i64 = @intCast(samples.len);
        const n_padded = @divTrunc(n_samples + fold - 1, fold) * fold;

        var mel: Mel = .{};
        mel.n_len = n_padded;
        mel.n_len_org = n_samples;
        mel.n_mel = 1;
        try mel.data.appendNTimes(gpa, 0.0, @intCast(n_padded));
        @memcpy(mel.data.items[0..samples.len], samples);

        for (output.items) |*m| m.deinit();
        output.clearRetainingCapacity();
        try output.append(gpa, mel);
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:1550 @b1b770d90), a member of `mtmd_audio_preprocessor_granite_speech`:
    /// HTK log10 mel, whisper normalization, consecutive frames stacked in pairs.
    fn graniteSpeech(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        const hp = self.hparams;
        if (samples.len == 0) return false;
        self.assertCache(1558);

        const n_fft = hp.audio_n_fft;
        const pad = @divTrunc(n_fft, 2);
        const ns: c_int = @intCast(samples.len);

        // reflect padding
        const n_padded: usize = samples.len + 2 * @as(usize, @intCast(pad));
        var padded: std.ArrayList(f32) = .empty;
        defer padded.deinit(gpa);
        try padded.appendNTimes(gpa, 0, n_padded);
        const pu: usize = @intCast(pad);
        @memcpy(padded.items[pu..][0..samples.len], samples);
        var i: c_int = 0;
        while (i < pad) : (i += 1) {
            var src = i + 1;
            if (src >= ns) src = ns - 1;
            padded.items[@intCast(pad - 1 - i)] = samples[@intCast(src)];
        }
        i = 0;
        while (i < pad) : (i += 1) {
            var src = ns - 2 - i;
            if (src < 0) src = 0;
            padded.items[@intCast(pad + ns + i)] = samples[@intCast(src)];
        }

        var mel: Mel = .{};
        defer mel.deinit();
        if (!try logMelSpectrogram(padded.items, &self.mel_config, &self.cache, shape_only, &mel)) return false;

        var mmax: f64 = -1e20;
        for (mel.data.items) |v| {
            if (v > mmax) mmax = v;
        }
        mmax -= 8.0;

        for (mel.data.items) |*v| {
            if (v.* < mmax) v.* = @floatCast(mmax);
            v.* = @floatCast((@as(f64, v.*) + 4.0) / 4.0);
        }

        var n_frames = mel.n_len;
        if (@rem(n_frames, 2) == 1) n_frames -= 1;
        const n_mel = mel.n_mel;
        const n_stacked = @divTrunc(n_frames, 2);

        var stacked: Mel = .{ .n_mel = 2 * n_mel, .n_len = n_stacked, .n_len_org = @intCast(samples.len) };
        try stacked.data.resize(gpa, 2 * @as(usize, @intCast(n_mel)) * @as(usize, @intCast(n_stacked)));

        var t: i64 = 0;
        while (t < n_stacked) : (t += 1) {
            var m: i64 = 0;
            while (m < n_mel) : (m += 1) {
                stacked.data.items[@intCast(m * n_stacked + t)] = mel.data.items[@intCast(m * mel.n_len + 2 * t)];
                stacked.data.items[@intCast((m + n_mel) * n_stacked + t)] = mel.data.items[@intCast(m * mel.n_len + 2 * t + 1)];
            }
        }

        try output.append(gpa, stacked);
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:1666 @b1b770d90), a member of `mtmd_audio_preprocessor_gemma4a`:
    /// 30 s chunks, semicausal padding, trimmed to PyTorch's frame count. The
    /// trim lowers `n_len` and leaves the data at its original stride, as
    /// the C++ does.
    fn gemma4a(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        const hp = self.hparams;
        if (samples.len == 0) return false;
        self.assertCache(1674);

        // Split into 30-second chunks (model context limit, ~750 tokens each)
        const chunk_samples: usize = @as(usize, gemma4a_segment_seconds) *% @as(usize, @bitCast(@as(isize, hp.audio_sample_rate)));
        var off: usize = 0;
        while (off < samples.len) : (off += chunk_samples) {
            const chunk_len = @min(chunk_samples, samples.len - off);
            const chunk_ptr = samples[off..][0..chunk_len];

            // Semicausal left-padding + right-padding to match PyTorch frame count
            const pad_left = @divTrunc(hp.audio_window_len, 2);
            const fft_size = hp.audio_n_fft;
            const hop = hp.audio_hop_len;
            const n_with_left: c_int = @as(c_int, @intCast(chunk_len)) + pad_left;
            // PyTorch: unfold(size=frame_length+1, step=hop) on semicausal-padded waveform
            const pt_frames: i64 = @divTrunc(n_with_left - (hp.audio_window_len + 1), hop) + 1;
            const n_padded_needed: i64 = (pt_frames - 1) * hop + fft_size;
            const total_pad: c_int = @max(@as(c_int, @intCast(n_padded_needed - @as(c_int, @intCast(chunk_len)))), pad_left);
            var padded_samples: std.ArrayList(f32) = .empty;
            defer padded_samples.deinit(gpa);
            try padded_samples.appendNTimes(gpa, 0, @as(usize, @intCast(total_pad)) + chunk_len);
            @memcpy(padded_samples.items[@intCast(pad_left)..][0..chunk_len], chunk_ptr);

            var out_chunk: Mel = .{};
            if (!try logMelSpectrogram(padded_samples.items, &self.mel_config, &self.cache, shape_only, &out_chunk)) {
                out_chunk.deinit();
                return false;
            }

            // Trim to PyTorch frame count
            out_chunk.n_len = @min(out_chunk.n_len, pt_frames);

            try output.append(gpa, out_chunk);
        }

        return true;
    }

    /// Ports `log_mel` (mtmd-audio.cpp:1740 @b1b770d90), a member of
    /// `mtmd_audio_preprocessor_parakeet`: the log-mel without normalization,
    /// into `out_full`, which the caller owns and frees on error too.
    fn parakeetLogMel(self: *const Preprocessor, samples_in: []const f32, out_full: *Mel, shape_only: bool) !bool {
        self.assertCache(1741);

        const frame_size: c_int = @intCast((self.mel_config.n_fft_bins - 1) * 2);

        // Apply preemphasis filter (high-pass): x[i] = x[i] - 0.97 * x[i-1]
        var pre: std.ArrayList(f32) = .empty;
        defer pre.deinit(gpa);
        try pre.appendSlice(gpa, samples_in);
        {
            const preemph: f32 = 0.97;
            var i: c_int = @as(c_int, @intCast(samples_in.len)) - 1;
            while (i > 0) : (i -= 1) {
                const iu: usize = @intCast(i);
                pre.items[iu] = @mulAdd(f32, -preemph, pre.items[iu - 1], pre.items[iu]);
            }
        }

        // Parakeet uses centered constant padding
        const pad: usize = @intCast(@divTrunc(frame_size, 2));
        var padded: std.ArrayList(f32) = .empty;
        defer padded.deinit(gpa);
        try padded.appendNTimes(gpa, 0, samples_in.len + 2 * pad);
        @memcpy(padded.items[pad..][0..samples_in.len], pre.items);

        if (!try logMelSpectrogram(padded.items, &self.mel_config, &self.cache, shape_only, out_full)) {
            return false;
        }
        out_full.n_len_org = out_full.n_len;
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:1768 @b1b770d90), a member of `mtmd_audio_preprocessor_parakeet`.
    fn parakeet(self: *const Preprocessor, samples_in: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        if (samples_in.len == 0) return false;

        var out_full: Mel = .{};
        const ok = self.parakeetLogMel(samples_in, &out_full, shape_only) catch |e| {
            out_full.deinit();
            return e;
        };
        if (!ok) {
            out_full.deinit();
            return false;
        }
        const frame_step = self.hparams.audio_hop_len;

        // Per-feature normalization (only on valid frames)
        {
            const eps: f64 = 1e-5;
            const valid_frames: c_int = @intCast(samples_in.len / @as(usize, @intCast(frame_step)));
            const vf: usize = @intCast(valid_frames);
            const n_len: usize = @intCast(out_full.n_len);

            var j: usize = 0;
            while (j < out_full.n_mel) : (j += 1) {
                const row = out_full.data.items[j * n_len ..][0..n_len];
                var sum: f64 = 0.0;
                var sq_diff_sum: f64 = undefined;

                // Calculate Mean ONLY on valid audio frames
                for (row[0..vf]) |v| sum += v;
                const mean = sum / @as(f64, @floatFromInt(valid_frames));

                // Calculate Variance ONLY on valid audio frames
                sq_diff_sum = strictSquareSum(row[0..vf], mean);

                const std_dev = @sqrt(sq_diff_sum / (@as(f64, @floatFromInt(valid_frames)) - 1.0));
                const denominator = std_dev + eps;

                // Apply to ALL frames (including the padded ones)
                for (row) |*v| v.* = @floatCast((@as(f64, v.*) - mean) / denominator);
            }
        }

        try output.append(gpa, out_full);
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:1820 @b1b770d90), a member of
    /// `mtmd_audio_preprocessor_nemotron_transcribe`: one chunk per
    /// recording, the raw log-mel of its valid frames.
    fn nemotronTranscribe(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel), shape_only: bool) !bool {
        const hp = self.hparams;
        if (samples.len < @as(usize, @intCast(hp.audio_n_fft))) return false;

        var mel: Mel = .{};
        const ok = self.parakeetLogMel(samples, &mel, shape_only) catch |e| {
            mel.deinit();
            return e;
        };
        if (!ok) {
            mel.deinit();
            return false;
        }

        // frames past n_samples / hop are padding (NeMo get_seq_len); kept raw, encode_windows normalizes them for ASR
        const n_frames: i64 = @intCast(samples.len / @as(usize, @intCast(hp.audio_hop_len)));
        const nf: usize = @intCast(n_frames);
        const n_len: usize = @intCast(mel.n_len);
        var j: usize = 1;
        while (j < mel.n_mel) : (j += 1) {
            std.mem.copyForwards(f32, mel.data.items[j * nf ..][0..nf], mel.data.items[j * n_len ..][0..nf]);
        }
        mel.n_len = n_frames;
        mel.n_len_org = n_frames;
        mel.data.shrinkRetainingCapacity(@as(usize, @intCast(mel.n_mel)) * nf);
        output.append(gpa, mel) catch |e| {
            mel.deinit();
            return e;
        };
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:1853 @b1b770d90), a member of `mtmd_audio_preprocessor_gemma4ua`:
    /// the raw waveform in 640-sample tokens, mel-major.
    fn gemma4ua(self: *const Preprocessor, samples: []const f32, output: *std.ArrayList(Mel)) !bool {
        const hp = self.hparams;
        if (samples.len == 0) return false;

        const frame_size = hp.n_mel_bins; // 640 samples per token @ 16 kHz = 40 ms
        const n_tokens = @divTrunc(@as(c_int, @intCast(samples.len)) + frame_size - 1, frame_size);

        var mel: Mel = .{ .n_len = n_tokens, .n_len_org = n_tokens, .n_mel = frame_size };
        const fs: usize = @intCast(frame_size);
        const nt: usize = @intCast(n_tokens);
        try mel.data.appendNTimes(gpa, 0, fs * nt);

        // stored mel-major (data[f * n_tokens + t]): the graph builder
        // transposes before RMSNorm so normalization is over frame_size
        for (0..nt) |t| {
            for (0..fs) |f| {
                const src = t * fs + f;
                mel.data.items[f * nt + t] = if (src < samples.len) samples[src] else 0.0;
            }
        }

        try output.append(gpa, mel);
        return true;
    }

    /// Ports `process` (mtmd-audio.cpp:1995 @b1b770d90), a member of `mtmd_audio_preprocessor_pockettts`:
    /// mimi convolves the raw 24 kHz waveform, so this only pads it to whole
    /// frames, as one single-row "mel".
    fn pockettts(self: *const Preprocessor, samples_in: []const f32, output: *std.ArrayList(Mel)) !bool {
        const hp = self.hparams;
        // the encoder needs whole frames, see pad_for_conv1d() in the reference
        const frame_size: i64 = @as(i64, hp.mimi_downsample) * 120;
        if (samples_in.len == 0 or frame_size <= 0) return false;

        var n_samples = samples_in.len;
        // the mimi transformer mask is dense, so cost is quadratic in the reference length
        const max_samples: i64 = @as(i64, Hparams.pockettts_max_spk_seconds) * hp.audio_sample_rate;
        if (@as(i64, @intCast(n_samples)) > max_samples) {
            impl.logWrn("%s: speaker reference is %.1f s, truncating to the first %d s\n", .{
                "process",
                @as(f64, @floatFromInt(n_samples)) / @as(f64, @floatFromInt(hp.audio_sample_rate)),
                @as(c_int, Hparams.pockettts_max_spk_seconds),
            });
            n_samples = @intCast(max_samples);
        }

        const n_frames: i64 = @divTrunc(@as(i64, @bitCast(@as(u64, n_samples) +% @as(u64, @bitCast(frame_size)) -% 1)), frame_size);
        const n_padded = n_frames * frame_size;

        var out: Mel = .{ .n_mel = 1, .n_len = n_padded, .n_len_org = @intCast(n_samples) };
        try out.data.appendNTimes(gpa, 0, @intCast(n_padded));
        @memcpy(out.data.items[0..n_samples], samples_in[0..n_samples]);

        try output.append(gpa, out);
        return true;
    }
};

/// Ports `mtmd_audio_streaming_istft` (mtmd-audio.h:266 @b1b770d90): STFT
/// frames back to audio, one frame at a time, overlap-added.
pub const StreamingIstft = struct {
    n_fft: c_int,
    hop_length: c_int,
    n_fft_bins: c_int,

    // Own cache for output processing
    cache: Cache = .{},

    // Streaming state
    overlap_buffer: []f32,
    window_sum_buffer: []f32,
    padding_to_remove: c_int,

    // Working buffers for IFFT
    ifft_in: []f32,
    ifft_out: []f32,

    /// Ports `mtmd_audio_streaming_istft` (mtmd-audio.cpp:1888 @b1b770d90), a member of `mtmd_audio_streaming_istft`.
    pub fn init(n_fft: c_int, hop_length: c_int) !StreamingIstft {
        const nu: usize = @intCast(n_fft);
        var self: StreamingIstft = .{
            .n_fft = n_fft,
            .hop_length = hop_length,
            .n_fft_bins = @divTrunc(n_fft, 2) + 1,
            .overlap_buffer = try gpa.alloc(f32, nu),
            .window_sum_buffer = try gpa.alloc(f32, nu),
            .padding_to_remove = @divTrunc(n_fft - hop_length, 2),
            .ifft_in = try gpa.alloc(f32, nu * 2 * 4), // extra space for recursive IFFT
            .ifft_out = try gpa.alloc(f32, nu * 2 * 4),
        };
        @memset(self.overlap_buffer, 0.0);
        @memset(self.window_sum_buffer, 0.0);
        @memset(self.ifft_in, 0.0);
        @memset(self.ifft_out, 0.0);
        ggmlAssert(n_fft > 0 and hop_length > 0 and hop_length <= n_fft, 1897, "n_fft > 0 && hop_length > 0 && hop_length <= n_fft");
        try self.cache.fillSinCosTable(@intCast(n_fft));
        try self.cache.fillHannWindow(@intCast(n_fft), true);
        return self;
    }

    pub fn deinit(self: *StreamingIstft) void {
        self.cache.deinit();
        gpa.free(self.overlap_buffer);
        gpa.free(self.window_sum_buffer);
        gpa.free(self.ifft_in);
        gpa.free(self.ifft_out);
    }

    /// Ports `reset` (mtmd-audio.cpp:1902 @b1b770d90), a member of `mtmd_audio_streaming_istft`.
    pub fn reset(self: *StreamingIstft) void {
        @memset(self.overlap_buffer, 0.0);
        @memset(self.window_sum_buffer, 0.0);
        self.padding_to_remove = @divTrunc(self.n_fft - self.hop_length, 2);
    }

    /// The normalized sample at `i` of the overlap buffer.
    fn sampleAt(self: *const StreamingIstft, i: usize) f32 {
        return if (self.window_sum_buffer[i] > 1e-8) self.overlap_buffer[i] / self.window_sum_buffer[i] else self.overlap_buffer[i];
    }

    /// Shifts both buffers left by `n`, zero-filling the end.
    fn shift(self: *StreamingIstft, n: usize) void {
        const len = self.overlap_buffer.len;
        std.mem.copyForwards(f32, self.overlap_buffer[0 .. len - n], self.overlap_buffer[n..]);
        @memset(self.overlap_buffer[len - n ..], 0.0);
        std.mem.copyForwards(f32, self.window_sum_buffer[0 .. len - n], self.window_sum_buffer[n..]);
        @memset(self.window_sum_buffer[len - n ..], 0.0);
    }

    /// Ports `process_frame` (mtmd-audio.cpp:1908 @b1b770d90), a member of `mtmd_audio_streaming_istft`:
    /// one frame, `n_fft_bins` interleaved real/imaginary pairs.
    ///
    /// Return: up to `hop_length` samples; the caller owns them.
    pub fn processFrame(self: *StreamingIstft, frame_spectrum: [*]const f32) !std.ArrayList(f32) {
        var output: std.ArrayList(f32) = .empty;
        try output.appendNTimes(gpa, 0, @intCast(self.hop_length));
        const bins: usize = @intCast(self.n_fft_bins);
        const nu: usize = @intCast(self.n_fft);

        // copy frequencies
        for (0..bins) |j| {
            self.ifft_in[j * 2 + 0] = frame_spectrum[j * 2 + 0];
            self.ifft_in[j * 2 + 1] = frame_spectrum[j * 2 + 1];
        }

        // mirror negative frequencies
        var j: usize = 1;
        while (j + 1 < bins) : (j += 1) {
            const mirror_idx = nu - j;
            self.ifft_in[mirror_idx * 2 + 0] = self.ifft_in[j * 2 + 0];
            self.ifft_in[mirror_idx * 2 + 1] = -self.ifft_in[j * 2 + 1]; // conjugate
        }

        ifft(&self.cache, self.ifft_in.ptr, self.n_fft, self.ifft_out.ptr);

        // update window sum and overlap buffer
        const hw = self.cache.hann_window.items;
        for (0..nu) |k| {
            self.window_sum_buffer[k] = @mulAdd(f32, hw[k], hw[k], self.window_sum_buffer[k]);
            self.overlap_buffer[k] = @mulAdd(f32, self.ifft_out[k * 2], hw[k], self.overlap_buffer[k]);
        }

        // extract hop_length samples with normalization
        for (output.items, 0..) |*o, i| o.* = self.sampleAt(i);

        // shift buffers left by hop_length
        self.shift(@intCast(self.hop_length));

        // Remove padding if needed
        const to_remove = @min(self.padding_to_remove, @as(c_int, @intCast(output.items.len)));
        self.padding_to_remove -= to_remove;
        output.replaceRangeAssumeCapacity(0, @intCast(to_remove), &.{});

        return output;
    }

    /// Ports `flush` (mtmd-audio.cpp:1956 @b1b770d90), a member of `mtmd_audio_streaming_istft`:
    /// what is left in the overlap buffer at the end of the stream.
    ///
    /// Return: the remaining samples; the caller owns them.
    pub fn flush(self: *StreamingIstft) !std.ArrayList(f32) {
        var output: std.ArrayList(f32) = .empty;
        var remaining = self.n_fft - self.hop_length;
        while (remaining > 0) {
            const chunk_size: usize = @intCast(@min(remaining, self.hop_length));
            for (0..chunk_size) |i| try output.append(gpa, self.sampleAt(i));
            self.shift(chunk_size);
            remaining -= @intCast(chunk_size);
        }
        return output;
    }
};

/// The files beside this one, for `harness/mtmd_audio_diff.zig`, which
/// builds this file as its own module root (`build/mtmd_audio_diff.zig`).
pub const deps = struct {
    pub const impl = @import("impl.zig");
    pub const model = @import("model.zig");
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the FFT of an impulse is flat" {
    var cache: Cache = .{};
    defer cache.deinit();
    try cache.fillSinCosTable(8);
    var in: [64]f32 = @splat(0);
    in[0] = 1.0;
    var out: [128]f32 = @splat(0);
    fft(&cache, &in, 8, &out);
    for (0..8) |k| {
        try std.testing.expectEqual(@as(f32, 1.0), out[2 * k]);
        try std.testing.expectEqual(@as(f32, 0.0), out[2 * k + 1]);
    }
}
