//! The flash-attention vector-kernel tuning table and the lookup over it.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml-metal/ggml-metal-tuning.cpp` — the logic
//! - `llama.cpp/ggml/src/ggml-metal/ggml-metal-tuning.h`   — the types and buckets
//!
//! both at v0.3.0 (`c1d0e7a00`). `tuning_table.zig` holds the 936
//! generated rows.
//!
//! # This file had to go first, and `PLAN.md` said otherwise
//!
//! Its seven entry points are the only **C++-linkage** symbols left in
//! ggml, and the plan recorded them as "called only by
//! `ggml-metal-ops.cpp`". Measured: `ggml-metal.o` references five of
//! them, `ggml-metal-ops.o` two, `ggml-metal-device.o` one — **three**
//! translation units. Every remaining Metal C++ file needs this one, so
//! either it goes first or all four move together as 7,509 live lines.
//!
//! # Zig supplies the mangled names
//!
//! `CLAUDE.md` records `ggml-backend-dl.cpp`'s three functions as having
//! "C++ linkage Zig cannot provide" — true of *those*, whose signatures
//! name `std::filesystem::path`. These seven are POD: ints, `int64_t`, an
//! enum, and a two-`int8_t` struct. So `@export` with the mangled name
//! works, and the three C++ callers keep linking against it unchanged
//! while each is ported in its own step.
//!
//! **One of the seven needs a workaround.** `fa_vec_set_override` takes
//! `fa_vec_cfg_t` **by value**, and Zig 0.16 hands a callee zeros for an
//! `extern struct` that size — see `CLAUDE.md`, and
//! `harness/abi_structs.c` for the measured table of shapes. The exported
//! thunk takes a `u16` and `@bitCast`s, which is the same two bytes in the
//! same register and was measured correct.

const std = @import("std");
const impl = @import("../impl.zig");
const table = @import("tuning_table.zig");

const c = impl.c;

pub const DeviceId = table.DeviceId;
pub const Cfg = table.Cfg;
pub const Key = table.Key;

/// Mirrors `FA_VEC_NE11_BUCKETS` (ggml-metal-tuning.h:15 @c1d0e7a00): KV
/// length buckets. The Q>1 crossover is head-size dependent, so the KV
/// axis is bucketed rather than thresholded once.
const fa_vec_ne11_buckets = [_]i32{ 1024, 4096, 16384 };

/// Mirrors `FA_VEC_NE01_BUCKETS` (ggml-metal-tuning.h:16 @c1d0e7a00):
/// query rows, splitting decode (== 1) from batch (>= 2).
const fa_vec_ne01_buckets = [_]i32{ 2, 3, 4, 5 };

/// Mirrors `FA_VEC_NE11_DEFAULT` (ggml-metal-tuning.h:30 @c1d0e7a00): the
/// `ne11_b` value a domain-default row carries.
const fa_vec_ne11_default: i8 = -1;
/// Mirrors `FA_VEC_DOMAIN_DECODE` (ggml-metal-tuning.h:31 @c1d0e7a00).
const fa_vec_domain_decode: i8 = 0;
/// Mirrors `FA_VEC_DOMAIN_BATCH` (ggml-metal-tuning.h:32 @c1d0e7a00).
const fa_vec_domain_batch: i8 = 1;

/// Ports `fa_vec_ne11_bucket` (ggml-metal-tuning.cpp:9 @c1d0e7a00).
///
/// Parameters:
/// - `ne11`: the KV length.
///
/// Return: the bucket index, or the bucket count when past the last edge.
fn ne11Bucket(ne11: i64) c_int {
    for (fa_vec_ne11_buckets, 0..) |edge, i| {
        if (ne11 < edge) return @intCast(i);
    }
    return fa_vec_ne11_buckets.len;
}

/// Ports `fa_vec_ne01_bucket` (ggml-metal-tuning.cpp:18 @c1d0e7a00).
///
/// Parameters:
/// - `ne01`: the query-row count.
///
/// Return: the bucket index, or the bucket count when past the last edge.
fn ne01Bucket(ne01: i64) c_int {
    for (fa_vec_ne01_buckets, 0..) |edge, i| {
        if (ne01 < edge) return @intCast(i);
    }
    return fa_vec_ne01_buckets.len;
}

/// Ports `fa_vec_baseline_ne` (ggml-metal-tuning.cpp:27 @c1d0e7a00): the
/// `NE` baked into each `(dk, dv)` baseline instantiation in
/// `kernels/fa.metal`.
///
/// The C writes ten `if`s and a default; this is the same table as a
/// `switch` on the pair. Hand-maintained there and here — it mirrors the
/// Metal source, which is never ported.
///
/// Parameters:
/// - `dk`: key head size.
/// - `dv`: value head size.
///
/// Return: the baseline `NE`, 4 when the pair has no instantiation.
fn baselineNe(dk: c_int, dv: c_int) c_int {
    return switch (dk) {
        32 => if (dv == 32) 4 else 4,
        64 => if (dv == 64) 2 else 4,
        96 => if (dv == 96) 4 else 4,
        128 => if (dv == 128) 1 else 4,
        192 => if (dv == 192 or dv == 128) 2 else 4,
        256 => if (dv == 256) 1 else 4,
        320 => if (dv == 256) 2 else 4,
        512 => if (dv == 512) 1 else 4,
        576 => if (dv == 512) 2 else 4,
        else => 4, // template default
    };
}

/// Ports `fa_vec_baseline_cfg` (ggml-metal-tuning.cpp:61 @c1d0e7a00).
fn baselineCfg(dk: c_int, dv: c_int) Cfg {
    return .{ .Q = 1, .NE = @intCast(baselineNe(dk, dv)) };
}

/// Ports `fa_vec_family_representative` (ggml-metal-tuning.cpp:1011
/// @c1d0e7a00): the SKU a GPU family falls back to.
fn familyRepresentative(gpu_family: c_int) DeviceId {
    return switch (gpu_family) {
        9 => .m4_max,
        else => .generic,
    };
}

var g_override_set: bool = false;
var g_override_cfg: Cfg = .{ .Q = 1, .NE = 4 };

/// Ports `find_cfg` (ggml-metal-tuning.cpp:1030 @c1d0e7a00).
///
/// The C compares keys with `memcmp` over all eight bytes, which its own
/// `static_assert(sizeof(fa_vec_key_t) == 8)` exists to license. The key
/// has no padding, so one `u64` comparison is the same test.
fn findCfg(k: Key) ?Cfg {
    const want: u64 = @bitCast(k);
    for (table.fa_vec_tuned_table) |entry| {
        if (@as(u64, @bitCast(entry.key)) == want) return entry.cfg;
    }
    return null;
}

/// Ports `fa_vec_pick`'s inner lambda `lookup`
/// (ggml-metal-tuning.cpp:1058 @c1d0e7a00): the exact bucket, then the
/// `ne01` domain default with `ne11` collapsed.
fn lookup(dev: DeviceId, dtype: c_int, dk: c_int, dv: c_int, ne11_b: c_int, ne01_b: c_int) ?Cfg {
    var k: Key = .{
        .device_id = @intCast(@intFromEnum(dev)),
        .dtype = @intCast(dtype),
        .dk = @intCast(dk),
        .dv = @intCast(dv),
        .ne11_b = @intCast(ne11_b),
        .ne01_b = @intCast(ne01_b),
    };
    if (findCfg(k)) |cfg| return cfg;

    k.ne11_b = fa_vec_ne11_default;
    k.ne01_b = if (ne01_b == 0) fa_vec_domain_decode else fa_vec_domain_batch;
    return findCfg(k);
}

/// Ports `fa_vec_pick` (ggml-metal-tuning.cpp:1039 @c1d0e7a00).
///
/// Parameters:
/// - `device_id`: the detected SKU.
/// - `gpu_family`: the Metal GPU family, 0 when unknown.
/// - `dtype`, `dk`, `dv`: the KV type and head sizes.
/// - `ne11`: KV length. - `ne01`: query rows.
///
/// Return: the `(Q, NE)` the FA vector kernel should be instantiated at.
fn pick(
    device_id: c_uint,
    gpu_family: c_int,
    dtype: c_int,
    dk: c_int,
    dv: c_int,
    ne11: i64,
    ne01: i64,
) Cfg {
    if (g_override_set) return g_override_cfg;

    const baseline = baselineCfg(dk, dv);

    const ne11_b = ne11Bucket(ne11);
    // short KV: attention is a small slice of the step, left to baseline
    if (ne11_b == 0) return baseline;

    const ne01_b = ne01Bucket(ne01);

    if (lookup(@enumFromInt(device_id), dtype, dk, dv, ne11_b, ne01_b)) |cfg| return cfg;

    // family fallback: retry under the family's representative SKU
    if (gpu_family > 0) {
        const rep = familyRepresentative(gpu_family);
        if (rep != .generic) {
            if (lookup(rep, dtype, dk, dv, ne11_b, ne01_b)) |cfg| return cfg;
        }
    }

    return baseline;
}

// -----------------------------------------------------------------------------
// The C++ ABI surface
//
// Seven symbols in `namespace ggml_metal_tuning`, exported under their
// mangled names so `ggml-metal.cpp`, `ggml-metal-ops.cpp` and
// `ggml-metal-device.cpp` link unchanged. Verified against `nm` on the
// reference object, not constructed by hand.

fn abiNe11Bucket(ne11: i64) callconv(.c) c_int {
    return ne11Bucket(ne11);
}
fn abiNe01Bucket(ne01: i64) callconv(.c) c_int {
    return ne01Bucket(ne01);
}
fn abiBaselineNe(dk: c_int, dv: c_int) callconv(.c) c_int {
    return baselineNe(dk, dv);
}
fn abiBaselineCfg(dk: c_int, dv: c_int) callconv(.c) Cfg {
    return baselineCfg(dk, dv);
}

/// `fa_vec_set_override(fa_vec_cfg_t)` takes the struct **by value**, which
/// Zig 0.16 receives as zeros. The thunk takes the same two bytes as a
/// `u16` instead; the caller is unchanged, since that is what the register
/// holds either way.
fn abiSetOverride(bits: u16) callconv(.c) void {
    g_override_cfg = @bitCast(bits);
    g_override_set = true;
}
fn abiClearOverride() callconv(.c) void {
    g_override_set = false;
}
fn abiPick(
    device_id: c_uint,
    gpu_family: c_int,
    dtype: c_int,
    dk: c_int,
    dv: c_int,
    ne11: i64,
    ne01: i64,
) callconv(.c) Cfg {
    return pick(device_id, gpu_family, dtype, dk, dv, ne11, ne01);
}

comptime {
    @export(&abiNe11Bucket, .{ .name = "_ZN17ggml_metal_tuning18fa_vec_ne11_bucketEx" });
    @export(&abiNe01Bucket, .{ .name = "_ZN17ggml_metal_tuning18fa_vec_ne01_bucketEx" });
    @export(&abiBaselineNe, .{ .name = "_ZN17ggml_metal_tuning18fa_vec_baseline_neEii" });
    @export(&abiBaselineCfg, .{ .name = "_ZN17ggml_metal_tuning19fa_vec_baseline_cfgEii" });
    @export(&abiSetOverride, .{ .name = "_ZN17ggml_metal_tuning19fa_vec_set_overrideENS_12fa_vec_cfg_tE" });
    @export(&abiClearOverride, .{ .name = "_ZN17ggml_metal_tuning21fa_vec_clear_overrideEv" });
    @export(&abiPick, .{ .name = "_ZN17ggml_metal_tuning11fa_vec_pickE20ggml_metal_device_idiiiixx" });
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the key is eight bytes with no padding, which the lookup depends on" {
    // The C's own `static_assert(sizeof(fa_vec_key_t) == 8, "must be
    // tightly packed for memcmp")`. `findCfg` compares one `u64`, which is
    // only the same test while that holds.
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Key));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(Key, "device_id"));
    try std.testing.expectEqual(@as(usize, 1), @offsetOf(Key, "dtype"));
    try std.testing.expectEqual(@as(usize, 2), @offsetOf(Key, "dk"));
    try std.testing.expectEqual(@as(usize, 4), @offsetOf(Key, "dv"));
    try std.testing.expectEqual(@as(usize, 6), @offsetOf(Key, "ne11_b"));
    try std.testing.expectEqual(@as(usize, 7), @offsetOf(Key, "ne01_b"));
}

test "the bucket edges are half-open on the low side" {
    try std.testing.expectEqual(@as(c_int, 0), ne11Bucket(0));
    try std.testing.expectEqual(@as(c_int, 0), ne11Bucket(1023));
    try std.testing.expectEqual(@as(c_int, 1), ne11Bucket(1024));
    try std.testing.expectEqual(@as(c_int, 3), ne11Bucket(16384));
    try std.testing.expectEqual(@as(c_int, 0), ne01Bucket(1));
    try std.testing.expectEqual(@as(c_int, 1), ne01Bucket(2));
    try std.testing.expectEqual(@as(c_int, 4), ne01Bucket(5));
}

test "a short KV always takes the baseline, whatever the table says" {
    // ne11 < FA_VEC_NE11_BUCKETS[0] short-circuits before any lookup.
    const got = pick(@intFromEnum(DeviceId.m4_max), 9, c.GGML_TYPE_F16, 128, 128, 512, 1);
    try std.testing.expectEqual(baselineCfg(128, 128), got);
}

test "the override wins over everything, and clearing restores the table" {
    defer abiClearOverride();
    const before = pick(@intFromEnum(DeviceId.m4_max), 9, c.GGML_TYPE_F16, 128, 128, 8192, 1);

    const forced: Cfg = .{ .Q = 7, .NE = 3 };
    abiSetOverride(@bitCast(forced));
    try std.testing.expectEqual(forced, pick(@intFromEnum(DeviceId.m4_max), 9, c.GGML_TYPE_F16, 128, 128, 8192, 1));

    abiClearOverride();
    try std.testing.expectEqual(before, pick(@intFromEnum(DeviceId.m4_max), 9, c.GGML_TYPE_F16, 128, 128, 8192, 1));
}

test "an unknown device falls back through its family, then to baseline" {
    // gpu_family 9 maps to m4_max; family 0 means unknown and cannot.
    const viaFamily = pick(@intFromEnum(DeviceId.generic), 9, c.GGML_TYPE_F16, 128, 128, 8192, 1);
    const direct = pick(@intFromEnum(DeviceId.m4_max), 0, c.GGML_TYPE_F16, 128, 128, 8192, 1);
    try std.testing.expectEqual(direct, viaFamily);

    const noFamily = pick(@intFromEnum(DeviceId.generic), 0, c.GGML_TYPE_F16, 128, 128, 8192, 1);
    try std.testing.expectEqual(baselineCfg(128, 128), noFamily);
}
