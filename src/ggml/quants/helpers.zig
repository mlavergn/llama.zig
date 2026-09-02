//! The scale-fitting routines the K-quants share.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-quants.c` (v0.3.0, `c1d0e7a00`), lines
//! 619-889. Each function names the C function it replaces and the line it
//! began at.
//!
//! # What these do
//!
//! A K-quant does not simply divide by the block maximum. It *searches* for the
//! scale that minimises weighted error over the block, trying a range of
//! candidates and keeping the best. That search is what `make_qx_quants` and
//! `make_qkx2_quants` are, and it is why quantizing a model is slow while
//! dequantizing is trivial.
//!
//! The weighting is the point of the importance matrix: `qw[i]` says how much
//! error in weight `i` costs, so the fit spends its precision where the model
//! is sensitive.
//!
//! # These are not approximations
//!
//! Every branch, every loop bound, every `is` in `-9..9` is reproduced exactly.
//! A "cleaner" search that converges to a similar scale produces different
//! bytes, and different bytes are a different model. The goldens compare bytes,
//! so any such improvement fails loudly -- which is the intended outcome.
//! # Float contraction, and why the goldens are built with it off
//!
//! **`zig cc` defaults to `-ffp-contract=on`, and so does Apple clang, but Zig
//! *language* code is strict.** The C fuses `a*b + c` into a single FMA -- one
//! rounding instead of two -- and a literal Zig translation does not. The
//! results differ in the last bit, which for a quantizer means different bytes
//! and therefore a different model file.
//!
//! Measured, not guessed: compiling `ggml-quants.c` with `-ffp-contract=off`
//! reproduces the strict-Zig bytes exactly; `on` or `fast` reproduces a stock
//! build's.
//!
//! Two ways out were tried:
//!
//! - **`@setFloatMode(.optimized)`** does not work. It permits contraction but
//!   LLVM does not then form the same FMAs.
//! - **Naming the sites with `@mulAdd`** works, but only if you guess exactly
//!   which expressions the compiler fuses and how it groups their operands.
//!   That guess held for the K-quants and then broke: fusing `makeQxQuants`'s
//!   accumulations fixed `q3_K` and simultaneously broke `q4_0`, which use the
//!   same function. Chasing it further would mean modelling a specific clang
//!   version's arithmetic across 5,000 lines.
//!
//! So `scripts/quants-golden` builds its reference with `-ffp-contract=off`
//! and the ported code stays plain. The comparison is still byte-exact and
//! still catches every real porting bug; both sides simply round the same way.
//!
//! **The consequence, stated plainly:** bytes these quantizers produce can
//! differ in the last bit from a stock llama.cpp build. That is invisible to
//! this project's deliverable -- our CLI reads model files and never writes
//! one -- but it would matter to anyone quantizing a model with it, and it is
//! a live question for Stage 3 steps 4 and 5, where float arithmetic *is* on
//! the inference path. See PLAN.md.

const std = @import("std");
const impl = @import("../impl.zig");
const c = impl.c;

/// Ports the `GROUP_MAX_EPS` family (ggml-quants.c:20 @c1d0e7a00).
///
/// The four members are consecutive: the plain one plus the `IQ3_XXS`,
/// `IQ2_S` and `IQ1_M` variants.
///
/// The threshold below which a block is treated as all zero and gets a zero
/// scale. `#define`s in the `.c` rather than the header, so they are not in the
/// import and have to be restated.
///
/// **The four values differ and the difference is load-bearing**: at 1e-15 a
/// block of denormals still gets fitted, while the i-quants bail out eight
/// orders of magnitude earlier. Using the general value for `iq3_xxs` would
/// send it down the fitting path on blocks the C rejects.
pub const group_max_eps: f32 = 1e-15;
pub const group_max_eps_iq3_xxs: f32 = 1e-8;
pub const group_max_eps_iq2_s: f32 = 1e-8;
pub const group_max_eps_iq1_m: f32 = 1e-7;
pub const group_max_eps_iq1_s: f32 = 1e-12;

/// Ports `nearest_int` (ggml-quants.c:621 @c1d0e7a00).
///
/// Round half to even, via the classic add-and-mask trick: adding
/// `12582912.0` (1.5 * 2^23) forces the value's mantissa into a fixed position,
/// after which the low bits *are* the integer.
///
/// **Not interchangeable with `@round`.** `@round` is half away from zero, this
/// is half to even, and they differ on exactly the ties a quantizer hits
/// constantly -- every weight that lands midway between two levels.
///
/// Parameters:
/// - `fval`: value to round. The C asserts `|fval| <= 4194303`, which the trick
///   requires; the assert is kept.
///
/// Return: the rounded integer.
pub inline fn nearestInt(fval: f32) i32 {
    std.debug.assert(@abs(fval) <= 4194303.0);
    const val = fval + 12582912.0;
    const i: i32 = @bitCast(val);
    return (i & 0x007fffff) - 0x00400000;
}

/// Ports `make_qx_quants` (ggml-quants.c:628 @c1d0e7a00).
///
/// Fits a symmetric scale for `n` weights quantized to `[-nmax, nmax-1]`,
/// searching 19 candidate scales around the naive one and keeping whichever
/// minimises weighted squared error.
///
/// Parameters:
/// - `n`: weights in the block.
/// - `nmax`: half the level count; levels run `[-nmax, nmax-1]`.
/// - `x`: the weights.
/// - `l_out`: receives the quantized levels, **biased by `nmax`** so they are
///   non-negative.
/// - `rmse_type`: selects the error weighting when `qw` is null -- 1 squares
///   the weight, 2 weights uniformly, 3 uses the magnitude, anything else its
///   square root. Zero skips the search entirely; negative returns after the
///   first fit.
/// - `qw`: optional per-weight importance, overriding `rmse_type`'s weighting.
///
/// Return: the chosen scale.
pub fn makeQxQuants(
    n: usize,
    nmax: i32,
    x: [*]const f32,
    l_out: [*]i8,
    rmse_type_in: i32,
    qw: ?[*]const f32,
) f32 {
    var max: f32 = 0;
    var amax: f32 = 0;
    for (0..n) |i| {
        const ax = @abs(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < group_max_eps) { // all zero
        for (0..n) |i| l_out[i] = 0;
        return 0.0;
    }

    var iscale = -@as(f32, @floatFromInt(nmax)) / max;
    if (rmse_type_in == 0) {
        for (0..n) |i| {
            const l = nearestInt(iscale * x[i]);
            l_out[i] = @intCast(nmax + @max(-nmax, @min(nmax - 1, l)));
        }
        return 1 / iscale;
    }

    var rmse_type = rmse_type_in;
    var return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }

    var sumlx: f32 = 0;
    var suml2: f32 = 0;
    for (0..n) |i| {
        var l = nearestInt(iscale * x[i]);
        l = @max(-nmax, @min(nmax - 1, l));
        l_out[i] = @intCast(l + nmax);
        const w = weightFor(x[i], rmse_type, qw, i);
        // Fused, as the C compiler fuses them. See the note at the top.
        sumlx += w * x[i] * @as(f32, @floatFromInt(l));
        suml2 += w * @as(f32, @floatFromInt(l)) * @as(f32, @floatFromInt(l));
    }
    var scale: f32 = if (suml2 != 0) sumlx / suml2 else 0.0;
    if (return_early) return if (suml2 > 0) 0.5 * (scale + 1 / iscale) else 1 / iscale;

    var best = scale * sumlx;
    // 19 candidates: the naive scale nudged by tenths in both directions.
    var is: i32 = -9;
    while (is <= 9) : (is += 1) {
        if (is == 0) continue;
        iscale = -(@as(f32, @floatFromInt(nmax)) + 0.1 * @as(f32, @floatFromInt(is))) / max;
        sumlx = 0;
        suml2 = 0;
        for (0..n) |i| {
            var l = nearestInt(iscale * x[i]);
            l = @max(-nmax, @min(nmax - 1, l));
            const w = weightFor(x[i], rmse_type, qw, i);
            sumlx += w * x[i] * @as(f32, @floatFromInt(l));
            suml2 += w * @as(f32, @floatFromInt(l)) * @as(f32, @floatFromInt(l));
        }
        // Cross-multiplied rather than dividing, so a zero suml2 cannot trap.
        if (suml2 > 0 and sumlx * sumlx > best * suml2) {
            for (0..n) |i| {
                const l = nearestInt(iscale * x[i]);
                l_out[i] = @intCast(nmax + @max(-nmax, @min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

/// The error weighting `make_qx_quants` applies, as a helper because the C
/// spells the same nested conditional out at three call sites.
inline fn weightFor(xi: f32, rmse_type: i32, qw: ?[*]const f32, i: usize) f32 {
    if (qw) |w| return w[i];
    return switch (rmse_type) {
        1 => xi * xi,
        2 => 1,
        3 => @abs(xi),
        else => @sqrt(@abs(xi)),
    };
}

/// Ports `make_q3_quants` (ggml-quants.c:697 @c1d0e7a00).
///
/// `q3_K`'s fit. Where `make_qx_quants` searches over scales, this holds the
/// scale and iterates the *levels*: five passes over the block, each trying to
/// move one weight to a better level and keeping the move only if it improves
/// the ratio `sumlx^2 / suml2`. It stops early once a pass changes nothing.
pub fn makeQ3Quants(n: usize, nmax: i32, x: [*]const f32, l_out: [*]i8, do_rmse: bool) f32 {
    var max: f32 = 0;
    var amax: f32 = 0;
    for (0..n) |i| {
        const ax = @abs(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < group_max_eps) { // all zero
        for (0..n) |i| l_out[i] = 0;
        return 0.0;
    }
    const iscale = -@as(f32, @floatFromInt(nmax)) / max;

    if (do_rmse) {
        var sumlx: f32 = 0;
        var suml2: f32 = 0;
        for (0..n) |i| {
            var l = nearestInt(iscale * x[i]);
            l = @max(-nmax, @min(nmax - 1, l));
            l_out[i] = @intCast(l);
            const w = x[i] * x[i];
            sumlx += w * x[i] * @as(f32, @floatFromInt(l));
            suml2 += w * @as(f32, @floatFromInt(l)) * @as(f32, @floatFromInt(l));
        }
        for (0..5) |_| {
            var n_changed: usize = 0;
            for (0..n) |i| {
                const w = x[i] * x[i];
                var slx = sumlx - w * x[i] * @as(f32, @floatFromInt(l_out[i]));
                if (slx > 0) {
                    var sl2 = suml2 - w * @as(f32, @floatFromInt(l_out[i])) * @as(f32, @floatFromInt(l_out[i]));
                    var new_l = nearestInt(x[i] * sl2 / slx);
                    new_l = @max(-nmax, @min(nmax - 1, new_l));
                    if (new_l != l_out[i]) {
                        slx += w * x[i] * @as(f32, @floatFromInt(new_l));
                        sl2 += w * @as(f32, @floatFromInt(new_l)) * @as(f32, @floatFromInt(new_l));
                        if (sl2 > 0 and slx * slx * suml2 > sumlx * sumlx * sl2) {
                            l_out[i] = @intCast(new_l);
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if (n_changed == 0) break;
        }
        // Bias into the unsigned range only at the end, because the search
        // above needs the signed level.
        for (0..n) |i| l_out[i] += @intCast(nmax);
        return if (suml2 > 0.0) sumlx / suml2 else 0.0;
    }

    for (0..n) |i| {
        var l = nearestInt(iscale * x[i]);
        l = @max(-nmax, @min(nmax - 1, l));
        l_out[i] = @intCast(l + nmax);
    }
    return 1 / iscale;
}

/// Ports `make_qkx2_quants` (ggml-quants.c:799 @c1d0e7a00).
///
/// The asymmetric fit, for the formats storing both a scale and a min. It tries
/// `nstep + 1` candidate scales and, for each, solves the weighted least
/// squares for the best `(scale, min)` pair together -- the `D` below is the
/// determinant of that 2x2 system.
///
/// Parameters:
/// - `n`, `nmax`: block size and top level.
/// - `x`: the weights.
/// - `weights`: per-weight importance; required, unlike `makeQxQuants`.
/// - `l_out`: receives the quantized levels.
/// - `the_min`: receives the fitted min, **negated** -- the caller stores
///   `-min` because the dequantizer adds it back.
/// - `l_aux`: scratch for the candidate levels; caller-provided so the search
///   allocates nothing.
/// - `rmin`, `rdelta`, `nstep`: the candidate range.
/// - `use_mad`: absolute error rather than squared.
///
/// Return: the chosen scale.
pub fn makeQkx2Quants(
    n: usize,
    nmax: i32,
    x: [*]const f32,
    weights: [*]const f32,
    l_out: [*]u8,
    the_min: *f32,
    l_aux: [*]u8,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) f32 {
    var min = x[0];
    var max = x[0];
    var sum_w = weights[0];
    var sum_x = sum_w * x[0];
    for (1..n) |i| {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        const w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    // The representable range always includes zero, so a block of entirely
    // positive weights still gets min = 0 rather than a tighter fit.
    if (min > 0) min = 0;
    if (max == min) {
        for (0..n) |i| l_out[i] = 0;
        the_min.* = -min;
        return 0.0;
    }

    var iscale = @as(f32, @floatFromInt(nmax)) / (max - min);
    var scale = 1 / iscale;
    var best_error: f32 = 0;
    // Two contraction sites below: the C fuses `scale*l + min` and
    // `best_error + weights[i]*diff`, and this does not. Fusing them with
    // `@mulAdd` was tried and reverted -- it fixed `q3_K` and broke `q4_0`,
    // which call this same function. See the note at the top of this file.
    for (0..n) |i| {
        const l = nearestInt(iscale * (x[i] - min));
        l_out[i] = @intCast(@max(0, @min(nmax, l)));
        var diff = scale * @as(f32, @floatFromInt(l_out[i])) + min - x[i];
        diff = if (use_mad) @abs(diff) else diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) {
        the_min.* = -min;
        return scale;
    }

    var is: i32 = 0;
    while (is <= nstep) : (is += 1) {
        iscale = (rmin + rdelta * @as(f32, @floatFromInt(is)) + @as(f32, @floatFromInt(nmax))) / (max - min);
        var sum_l: f32 = 0;
        var sum_l2: f32 = 0;
        var sum_xl: f32 = 0;
        for (0..n) |i| {
            var l = nearestInt(iscale * (x[i] - min));
            l = @max(0, @min(nmax, l));
            l_aux[i] = @intCast(l);
            const w = weights[i];
            const lf = @as(f32, @floatFromInt(l));
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        const d = sum_w * sum_l2 - sum_l * sum_l;
        if (d > 0) {
            var this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            var this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            // A positive min is not representable, so fall back to a pure
            // scale fit rather than clamping and keeping a now-wrong scale.
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            var cur_error: f32 = 0;
            for (0..n) |i| {
                var diff = this_scale * @as(f32, @floatFromInt(l_aux[i])) + this_min - x[i];
                diff = if (use_mad) @abs(diff) else diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                for (0..n) |i| l_out[i] = l_aux[i];
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    the_min.* = -min;
    return scale;
}

/// Ports `make_qkx3_quants` (ggml-quants.c:993 @c1d0e7a00).
///
/// `makeQkx2Quants`'s sibling, with two differences that matter:
///
/// - **`weights` is optional here.** When null the weighting falls back to
///   `x[i]*x[i]`, recomputed at each of the four places it is needed rather
///   than hoisted -- kept that way because hoisting changes nothing but makes
///   the correspondence with the C harder to check.
/// - **The degenerate test is `max <= min`, not `max == min`**, and it zeroes
///   `L` with a `memset` rather than a loop. Reachable when every weight is
///   negative: `min` is clamped to 0 above, so `max` can end up below it.
///
/// The search is otherwise the same weighted least-squares fit.
pub fn makeQkx3Quants(
    n: usize,
    nmax: i32,
    x: [*]const f32,
    weights: ?[*]const f32,
    l_out: [*]u8,
    the_min: *f32,
    l_aux: [*]u8,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) f32 {
    const wAt = struct {
        inline fn f(w: ?[*]const f32, xs: [*]const f32, i: usize) f32 {
            return if (w) |ws| ws[i] else xs[i] * xs[i];
        }
    }.f;

    var min = x[0];
    var max = x[0];
    var sum_w = wAt(weights, x, 0);
    var sum_x = sum_w * x[0];
    for (1..n) |i| {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        const w = wAt(weights, x, i);
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max <= min) {
        @memset(l_out[0..n], 0);
        the_min.* = -min;
        return 0.0;
    }

    var iscale = @as(f32, @floatFromInt(nmax)) / (max - min);
    var scale = 1 / iscale;
    var best_mad: f32 = 0;
    for (0..n) |i| {
        const l = nearestInt(iscale * (x[i] - min));
        l_out[i] = @intCast(@max(0, @min(nmax, l)));
        var diff = scale * @as(f32, @floatFromInt(l_out[i])) + min - x[i];
        diff = if (use_mad) @abs(diff) else diff * diff;
        best_mad += wAt(weights, x, i) * diff;
    }
    if (nstep < 1) {
        the_min.* = -min;
        return scale;
    }

    var is: i32 = 0;
    while (is <= nstep) : (is += 1) {
        iscale = (rmin + rdelta * @as(f32, @floatFromInt(is)) + @as(f32, @floatFromInt(nmax))) / (max - min);
        var sum_l: f32 = 0;
        var sum_l2: f32 = 0;
        var sum_xl: f32 = 0;
        for (0..n) |i| {
            var l = nearestInt(iscale * (x[i] - min));
            l = @max(0, @min(nmax, l));
            l_aux[i] = @intCast(l);
            const w = wAt(weights, x, i);
            const lf = @as(f32, @floatFromInt(l));
            sum_l += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        const d = sum_w * sum_l2 - sum_l * sum_l;
        if (d > 0) {
            var this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            var this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            var mad: f32 = 0;
            for (0..n) |i| {
                var diff = this_scale * @as(f32, @floatFromInt(l_aux[i])) + this_min - x[i];
                diff = if (use_mad) @abs(diff) else diff * diff;
                mad += wAt(weights, x, i) * diff;
            }
            if (mad < best_mad) {
                for (0..n) |i| l_out[i] = l_aux[i];
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    the_min.* = -min;
    return scale;
}

/// Ports `make_qp_quants` (ggml-quants.c:1076 @c1d0e7a00).
///
/// Fits a scale for values that are all **non-negative** -- the sub-block
/// scales and mins the K-quants have already produced. Hence `qp`: no sign to
/// worry about, so the levels run `0..nmax` rather than straddling zero.
///
/// Two stages: a nine-candidate scale search minimising weighted squared
/// error, then up to five refinement passes moving individual levels, the same
/// shape as `makeQ3Quants`'s.
///
/// Parameters:
/// - `n`, `nmax`: how many values and the top level.
/// - `x`: the values, all non-negative.
/// - `l_out`: receives the quantized levels.
/// - `quant_weights`: per-value importance; **required**, unlike the other
///   fits here.
///
/// Return: the chosen scale.
pub fn makeQpQuants(n: usize, nmax: i32, x: [*]const f32, l_out: [*]u8, quant_weights: [*]const f32) f32 {
    var max: f32 = 0;
    for (0..n) |i| max = @max(max, x[i]);

    if (max < group_max_eps) { // all zero
        for (0..n) |i| l_out[i] = 0;
        return 0.0;
    }

    var iscale = @as(f32, @floatFromInt(nmax)) / max;
    // Unclamped, and stored into a `uint8_t` in the C -- so a level above 255
    // wraps rather than saturating. `@intCast` would trap here; the wrap is
    // the behaviour being reproduced. The refinement passes below clamp
    // properly, so this only affects the initial guess.
    for (0..n) |i| l_out[i] = @truncate(@as(u32, @bitCast(nearestInt(iscale * x[i]))));
    const scale = 1 / iscale;

    var best_mse: f32 = 0;
    for (0..n) |i| {
        const diff = x[i] - scale * @as(f32, @floatFromInt(l_out[i]));
        best_mse += quant_weights[i] * diff * diff;
    }

    var is: i32 = -4;
    while (is <= 4) : (is += 1) {
        if (is == 0) continue;
        const iscale_is = (0.1 * @as(f32, @floatFromInt(is)) + @as(f32, @floatFromInt(nmax))) / max;
        const scale_is = 1 / iscale_is;
        var mse: f32 = 0;
        for (0..n) |i| {
            var l = nearestInt(iscale_is * x[i]);
            l = @min(nmax, l);
            const diff = x[i] - scale_is * @as(f32, @floatFromInt(l));
            mse += quant_weights[i] * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    var sumlx: f32 = 0;
    var suml2: f32 = 0;
    for (0..n) |i| {
        var l = nearestInt(iscale * x[i]);
        // `MIN(nmax, l)` only bounds it above; the C then narrows to
        // `uint8_t`, so a negative level wraps. Same reasoning as the store
        // above.
        l = @min(nmax, l);
        l_out[i] = @truncate(@as(u32, @bitCast(l)));
        const w = quant_weights[i];
        sumlx += w * x[i] * @as(f32, @floatFromInt(l));
        suml2 += w * @as(f32, @floatFromInt(l)) * @as(f32, @floatFromInt(l));
    }

    for (0..5) |_| {
        var n_changed: usize = 0;
        for (0..n) |i| {
            const w = quant_weights[i];
            var slx = sumlx - w * x[i] * @as(f32, @floatFromInt(l_out[i]));
            var sl2 = suml2 - w * @as(f32, @floatFromInt(l_out[i])) * @as(f32, @floatFromInt(l_out[i]));
            if (slx > 0 and sl2 > 0) {
                var new_l = nearestInt(x[i] * sl2 / slx);
                new_l = @min(nmax, new_l);
                if (new_l != l_out[i]) {
                    slx += w * x[i] * @as(f32, @floatFromInt(new_l));
                    sl2 += w * @as(f32, @floatFromInt(new_l)) * @as(f32, @floatFromInt(new_l));
                    // Cross-multiplied so neither side needs a division.
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        l_out[i] = @truncate(@as(u32, @bitCast(new_l)));
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if (n_changed == 0) break;
    }

    return if (suml2 > 0.0) sumlx / suml2 else 0.0;
}

/// Ports `get_scale_min_k4` (ggml-quants.c:880 @c1d0e7a00).
///
/// Unpacks one sub-block's 6-bit scale and 6-bit min from `q4_K`/`q5_K`'s
/// 12-byte `scales` array. The first four sub-blocks are stored plainly in the
/// low six bits; the last four have their low four bits in one byte and their
/// top two borrowed from the high bits of an earlier one.
pub inline fn getScaleMinK4(j: usize, q: [*]const u8, d: *u8, m: *u8) void {
    if (j < 4) {
        d.* = q[j] & 63;
        m.* = q[j + 4] & 63;
    } else {
        d.* = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m.* = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "nearestInt rounds half to even, not half away from zero" {
    // The distinction @round would get wrong, and the reason this exists.
    try std.testing.expectEqual(@as(i32, 0), nearestInt(0.5));
    try std.testing.expectEqual(@as(i32, 2), nearestInt(1.5));
    try std.testing.expectEqual(@as(i32, 2), nearestInt(2.5));
    try std.testing.expectEqual(@as(i32, 4), nearestInt(3.5));
    try std.testing.expectEqual(@as(i32, 0), nearestInt(-0.5));
    try std.testing.expectEqual(@as(i32, -2), nearestInt(-1.5));

    // @round disagrees on every one of those ties.
    try std.testing.expectEqual(@as(i32, 1), @as(i32, @intFromFloat(@round(@as(f32, 0.5)))));

    // Away from ties the two agree.
    try std.testing.expectEqual(@as(i32, 3), nearestInt(3.2));
    try std.testing.expectEqual(@as(i32, 4), nearestInt(3.7));
    try std.testing.expectEqual(@as(i32, -3), nearestInt(-3.2));
    try std.testing.expectEqual(@as(i32, 0), nearestInt(0.0));
    try std.testing.expectEqual(@as(i32, 1000), nearestInt(1000.0));
}

test "an all-zero block returns a zero scale rather than dividing by it" {
    var x: [16]f32 = @splat(0.0);
    var l: [16]i8 = undefined;
    try std.testing.expectEqual(@as(f32, 0.0), makeQxQuants(16, 8, &x, &l, 1, null));
    for (l) |v| try std.testing.expectEqual(@as(i8, 0), v);

    try std.testing.expectEqual(@as(f32, 0.0), makeQ3Quants(16, 4, &x, &l, true));
}

test "getScaleMinK4 splits the packed six-bit pairs" {
    // Sub-blocks 0-3 sit in the low six bits; 4-7 borrow their top two bits
    // from the high bits of bytes 0-3.
    var q: [12]u8 = @splat(0);
    q[0] = 63; // scale 0
    q[4] = 40; // min 0
    try blk: {
        var d: u8 = undefined;
        var m: u8 = undefined;
        getScaleMinK4(0, &q, &d, &m);
        break :blk std.testing.expectEqual(@as(u8, 63), d) catch |e| e;
    };

    var d: u8 = undefined;
    var m: u8 = undefined;
    getScaleMinK4(0, &q, &d, &m);
    try std.testing.expectEqual(@as(u8, 40), m);

    // The borrowed-bit path: bits 6-7 of q[0] become bits 4-5 of scale 4.
    q[0] = 0xC0; // top two bits set
    q[8] = 0x0F;
    getScaleMinK4(4, &q, &d, &m);
    try std.testing.expectEqual(@as(u8, 0x0F | (0x3 << 4)), d);
}
